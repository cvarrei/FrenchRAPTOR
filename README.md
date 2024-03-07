# FrenchRAPTOR - Make sure your RAG captures the essence of large documents [Mixtral-8x7B]

Please find the Medium link here: 


**MEDIUM ARTICLE:**
<p align="center">
<img src="/images/dalle.png" width="500" title="hover text">
</p>

In conventional Retrieval-Augmented Generation (RAG) architectures, textual content is segmented into multiple chunks/fragments, which are then matched to a user's query through similarity metrics such as cosine similarity to retrieve relevant segments. However, this approach encounters limitations when we try to capture the whole idea of extensive documents, such as entire books. For instance, considering a voluminous text like "Harry Potter," which comprises approximately 257,000 words, the challenge magnifies. Given Mixtral's context window capacity of 32,000 tokens, and the cost implications associated with querying models like OpenAI's GPT-4 - priced at $30.00 / 1M tokens (input)  - the financial and computational feasibility becomes questionable ! Imagine at the scale of 10,000 daily users and the need to analyze 100 large documents… !! Gemini-1.5 has a 1 million token context; perhaps that's the solution. But there's going to be a tonne of noise in the prompt! Imagine scanning the entire text for a certain concept or just attempting to higlight the most important ideas.

## FrenchRAPTOR 
Reading the paper made me think…To optimise the extraction of information from my documents, particularly the larger ones, I need to work on a simple solution that I could add to my processing pipelines and which can work fast.

I began to consider which model embedding to use. I preferred one that was light, but which has been shown to be efficient. By coincidence, the MTEB for French was recently published by Lyon-NLP members on HuggingFace (https://huggingface.co/blog/lyon-nlp-group/french-mteb-datasets , https://huggingface.co/spaces/mteb/leaderboard). Actually, might be not so original but I thought of using pre-trained models from Sentence-Transformers (both less than 200 MB): 

* all-MiniLM-L12-v2 : With 384 embedding dimensions, it ranks 24th overall in the French language (31st for clustering datasets: 33.75).
* all-MiniLM-L6-v2: For 384 embedding dimensions, it ranks 50th overall in the French language (but 26th for clustering datasets with 1-point difference: 34.73)..

When comparing the average performance on clustering datasets between these two (33.75 vs. 34.73) to the difference with the top-ranked model for clustering (49.67, OpenAI's text-embedding-ada-002), the difference appears to be not so important. However, all-MiniLM-L12-v2 shows a better overall performance than all-MiniLM-L6-v2.
For this reason, I decided to use all-MiniLM-L12-v2 for my embedding.


The authors used Gaussian Mixture Models to produce the clustering after performing a dimensionality reduction with UMAP. As a result, I questioned whether dimensionality reduction was necessary because my embedding dimensionality was smaller. I choose to skip this step to guarantee that no information is lost. 
Finally, as previously said, I simply want to expand my first corpus with more abstracted ideas so that if the user queries for more abstracted information, the RAG can quickly collect tiny summaries of more abstractive thoughts. As a result, I provide a hierarchical k-means framework (hard clustering). This is another significant variation from the original paper, which used soft clustering and a Gaussian Mixture Model. The cluster number with the highest average Silhouette score was then picked as the number of clusters to keep: for each point, the Silhouette score is the difference between the average distance to points in the same group as it (intra-cluster) and the average distance to points in other neighbouring groups (inter-clusters) - we use the average of all points.


Ascending clusterization implemented in FrenchRAPTORThe preceding graphic shows that the hierarchical clustering technique underpinning FrenchRAPTOR is built in an ascending order from specific details to broad themes. RAPTOR was created in a decreasing order, beginning with broad themes (global clusters), then divint into local clusters (specific details).
Sometimes, the generative model summarizes a single cluster of unique concepts as a numerical list (much like subclusters within a bigger cluster). If it was the case, I processed the items of the list and handled them as separate summaries that all the information was captured (adding them as different summaries). 

<img src="/images/differences.png" title="hover text">

Finally, the original article used OpenAI's gpt-3.5-turbo for the LLM-based summarization. In FrenchRAPTOR, I prefered an open-source model, MistralAI's Mixtral-8x7B.

## Let's put it in practice… Someone understood the European AIAct? 

The AIAct is a proposal for a European law on artificial intelligence (https://artificialintelligenceact.eu/fr/). I extract all the articles from the AiAct and summarized each one of them (to start with clean documents). The whole final document processed by FrenchRAPTOR has 20 984 tokens for 71 articles.

If the user wants to ask the general question: What is the AIAct about? This information is not present in one chunks, this is the interpretation of the overall document. If I implement the original articles as corpus for my RAG, I get the following response: **"The AiAct, or the Artificial Intelligence Regulation of the European Union, concerns the declaration of conformity, the CE conformity marking, and the designation of competent national authorities for high-risk AI systems in the EU."**
It's evident that not all the information is included in the response. It concentrates on high-risk AI systems and only partially reflects the content of the AIAct (the different chunks obtained, k=3). This answer is insufficient to address the user's question.


Let's start our FrenchRAPTOR algorithm, I started with a range of 3 to 24 clusters. Before generating the general summary of this final iteration, I applied the arbitrary cutoff of having four clusters or fewer at an iteration.

<img src="/images/silhouette.png" title="hover text">

By examining the above figures, we can observe that the FrenchRAPTOR method required three iterations in order to terminate the iterations after reaching the arbitrary criterion of fewer than five clusters. The stages are 7, 6, (augmented to 12), and 3 clusters, in that order (the number of clusters that maximized the the Silhouette score [black large point]).
Subsequently, the final three summaries were combined into the comprehensive overview. Thus, we ended up with four levels of abstractive summaries from the overall document, which we used to build a FrenchRAPTOR-enriched corpus using the collapsed tree approach (Sarthi et al., 2024).

<img src="/images/corpus.png" width="500" title="hover text">

Let's get into some of the summaries:

* The Document's overall Concept is: **"The AIAct is a European regulation that defines the rules governing artificial intelligence, prohibits certain practices, creates a European AI Committee, establishes rules for the marketing and use of AI systems, and defines a regulatory framework for high-risk systems, while encouraging voluntary codes of conduct and safeguarding intellectual property rights, confidentiality and the rights of defence. "**,
  
It's clear how this summary presents the main points of the document.

* While Summary 1.3, for example, is: **"The regulation establishes rules for the placing on the market and use of AI systems in the EU, prohibiting certain practices and imposing specific requirements for high-risk systems, such as technical documentation, registration, regulatory compliance, conformity assessment, record keeping, corrective action, notification of competent authorities, and affixing of the CE mark. Users must also comply with certain obligations regarding use and transparency. Measures are provided for small suppliers and users of AI, and a committee provides advice and assistance to the Commission in implementing the regulation. An EU safeguard procedure is also provided for in the event of measures taken by a Member State concerning a non-compliant AI system."**

This summary is much more specific about policies about high-risk AI systems. 

So, let's ask again our question: What is the AIAct about? We get now the following response: **"The AIAct is a European regulation that defines the rules for artificial intelligence, prohibiting certain practices, creating a European AI Committee, establishing rules for the marketing and use of AI systems, and defining a regulatory framework for high-risk systems."**

We can see that the response is more complete than the previous one: it addresses more of the themes highlighted in the various articles and accurately states that it is a law for all AI systems, not just high-risk ones. Actually, it is completely inspired by our Document's Overall Concept. 

## Conclusion 
Is it equivalent or superior than the original RAPTOR architecture? This is still an open subject; In the futur, I have planned to compare performance and resources (computational and time) to guarantee that a simpler version will work as well. Stay tuned ! 
However, what I have been able to assess is that it presents effectiveness and realistic processing times. Thus, despite the need for further work to make a robust evaluation performance, FrenchRAPTOR demonstrates that it is an effective solution for hierarchical abstraction of huge texts.

*Disclaimer: This algorithm has been developped and tested with French Documents, thus, the choice of models (generative or embedding models) might need to be changed accoridng to your language.*
