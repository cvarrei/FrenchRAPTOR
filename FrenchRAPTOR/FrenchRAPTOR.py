from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import re 
from itertools import chain
from typing import Callable, List, Dict, Union


class FrenchRAPTOR():
    def __init__(self, 
                 embedding_function: Callable,  # A function of embedding of an instantiated model
                 model_llm: str, # The HuggingFace repository of the generative model
                 context: str) -> None: # The context of the document (e.g., AIact, a European AI regulation document)
        self.encode = embedding_function
        self.llm = HuggingFaceHub(repo_id=model_llm,  model_kwargs={"temperature": 0.3, "max_new_tokens": 500})
        self.chain_raptor()
        self.chain_raptor_summaryfinal(context=context) # Create the final summary with the context of the document
        self.plot_data = []  # This will store the silhouette scores and cluster numbers for each iteration

    def chain_raptor(self) -> None:
        # Construt the chain of the different iterations.
        prompt_raptor =  """
        [INST]Résumes de manière générale les article de loi suivants sans rentrer dans les détails:
        {sentences_cluster}
        Limites toi à une seule phrase de résumé.
        [/INST] 

        Ces articles peuvent se résumer par la phrase suivante:
        """
        prompt_raptor_temp1 = PromptTemplate(input_variables=["sentences_cluster"], template=prompt_raptor)
        self.chainraptor_iter = LLMChain(prompt=prompt_raptor_temp1, llm=self.llm)

    def chain_raptor_summaryfinal(self, 
                                  context: str) -> None :
        # Construct the chain of the final summary
        prompt_raptor_final =  """
        [INST] Résumes de manière générale de quoi parle ce document,""" + context + """,sans rentrer dans les détails:
        {sentences_cluster}
        Limites toi à une seule phrase de résumé.
        [/INST] 
        Ces articles peuvent se résumer par la phrase suivante:
        """

        print(prompt_raptor_final)
        prompt_raptor_temp2 = PromptTemplate(input_variables=["sentences_cluster"], template=prompt_raptor_final)
        self.chainraptor_final = LLMChain(prompt=prompt_raptor_temp2, llm=self.llm)

    def fit(self,
            chunks: List[str],
            cluster_init_max: int = 25) -> Union[Dict[str, List[str]], List[str]]:
        """
        Fit the RAPTOR algorithm using KMeans hard clustering on the embeddings of the given chunks of text.

        This method iteratively clusters the text chunks, selects the best number of clusters based on silhouette scores,
        and generates summaries for each cluster. The process is repeated with the summaries, until we reach less than 4 clusters.
        Finally, a global summary is generated from the last set of chunks. It allows to have a hierarchical structure of summaries.

        Parameters:
        - cluster_init_max (int): The maximum initial number of clusters to consider for the first iteration.
        - chunks (List[str]): The initial chunks of text to be clustered and summarized.

        Returns:
        - Dict[str, List[str]]: A dictionary of hierarchical summaries
        - List[str]: The final corpus containing all generated summaries, including the global summary.

        """
        # We initialize the corpus
        corpus = chunks
        embeddings = self.encode(chunks) # We get the embedding of our documents 
        cluster_chosen = cluster_init_max # We initialize the maximum of the range of the initial clustering
        d_resume = {}
        i = 0 # Initialize the level counter

        # Loop to build hierarchical summaries until the number of clusters is less than 5
        while cluster_chosen > 5:
            i += 1
            label = f'Résumé de Niveau {i}'
            d_resume[label] = []  # Initialize list for storing summaries at the current level
            d = {"n_cluster": [],
                "silhouette": []}
            range_n_clusters = range(3,cluster_chosen)

            # Loop over possible cluster numbers to find the best one based on silhouette score
            for n_clusters in range_n_clusters:
                clusterer = KMeans(n_clusters=n_clusters, random_state=10, n_init='auto') # Hard clustering
                clusterer.fit(embeddings) # Clustering on the embeddings
                d["n_cluster"].append(n_clusters) 
                d["silhouette"].append(silhouette_score(embeddings, clusterer.labels_))

            self.plot_data.append({'n_cluster': d["n_cluster"], 'silhouette': d["silhouette"]})

            # Determine the optimal number of clusters with the highest silhouette score
            index_of_highest_silhouette = d["silhouette"].index(max(d["silhouette"]))
            cluster_chosen = d["n_cluster"][index_of_highest_silhouette]

            print(f"\n{cluster_chosen} Clusters choisis pour les résumés de niveau {i} \n")
            # Perform clustering with the optimal number of clusters
            km_clusers = KMeans(n_clusters=cluster_chosen, random_state=10, n_init='auto').fit(embeddings)
            pred2 = km_clusers.predict(embeddings)
            
            # Generate summaries for each cluster
            list_resume = []
            for j in range(0, cluster_chosen):
                print("\n Résumé de niveau -", i," Cluster ", j, "\n")
                cluster_i = np.array(chunks)[pred2 == j]
                summary_raptor = self.chainraptor_iter.invoke({"sentences_cluster": cluster_i})
                answer = summary_raptor["text"].strip()
                print(answer)
                list_resume.append(answer)
            # Process summaries to split and clean them (e.g., "1. The AIact....")
            list_resume = [re.split(r'\d+. ', document) for document in list_resume]
            list_resume = list(chain.from_iterable(list_resume)) # Flatten the list
            list_resume = [document for document in list_resume if len(document) > 2] # Remove empty items


            d_resume[label] = list_resume
            corpus = list(corpus) + list(list_resume) # Convert to list to ensure concatenation works as expected
            chunks = list_resume  # Use summaries as new chunks for the next iteration
            embeddings = self.encode(chunks) # Re-encode the new chunks

            if len(chunks) > cluster_init_max:
                cluster_chosen = cluster_init_max # To adapt range to current number of chunks after cleaning
            else:
                cluster_chosen = len(chunks) # To adapt range to current number of chunks after cleaning

        # Generate the global summary from the final set of summarie
        summary_raptor = self.chainraptor_final.invoke({"sentences_cluster": chunks})
        resume_global = summary_raptor["text"].strip()
        corpus = list(corpus) + [resume_global]
        d_resume["Résumé Global"] = [resume_global]

        print("\n Résumé Global du document \n")
        print(resume_global)
        
        return d_resume, corpus

    def plot_silhouette(self):
        """Generates and displays subplots for the silhouette scores vs. the number of clusters for each iteration."""
        num_iterations = len(self.plot_data)
        if num_iterations == 0:
            print("Raptor n'a pas encore été lancé, veuillez commencez par utilisez la fonction fit() sur vos documents.")
            return
        
        # Calculate the number of rows required for subplots based on the number of iterations
        nrows = int(np.ceil(num_iterations / 2))
        fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(15, 5 * nrows))
        axes = axes.flatten()  # Flatten the axes array for easy indexing
        
        for i, data in enumerate(self.plot_data):
            df_kmeans = pd.DataFrame({
                'n_cluster': data['n_cluster'],
                'silhouette': data['silhouette']
            })

            max_ind = np.argmax(df_kmeans["silhouette"])
            max_silhouette = df_kmeans["silhouette"][max_ind]
            max_cluster = df_kmeans["n_cluster"][max_ind]
            
            # Plotting on the i-th subplot
            axes[i].set_xticks(np.arange(df_kmeans['n_cluster'].min(), df_kmeans['n_cluster'].max() + 1, 1))
            sns.lineplot(data=df_kmeans, x="n_cluster", y="silhouette", color='#ff4255', ax=axes[i])
            sns.scatterplot(data=df_kmeans, x="n_cluster", y="silhouette", color='#ff4255', ax=axes[i], s=60)
            axes[i].plot(max_cluster, max_silhouette, "ko", ms = 10)
            axes[i].grid(axis='x')
            axes[i].set_title(f"Iteration {i+1}: Silhouette Score", fontsize=14)
            axes[i].set_xlabel("Nombre de clusters")
            axes[i].set_ylabel("Silhouette Score")
            
        # If the number of iterations is odd, hide the last subplot if it's unused
        if num_iterations % 2 != 0:
            axes[-1].axis('off')

        plt.tight_layout()
        plt.show()