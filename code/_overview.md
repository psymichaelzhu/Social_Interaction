Here’s a concise outline for your proposal, reflecting how you plan to link deep-learning–extracted semantic features from social interaction videos with neural response similarity:

⸻

1. Data & Motivation
	•	Data: 244 social-interaction video clips, along with simultaneous neural recordings while participants watch these videos.
	•	Goal: Identify which dimensions of social interaction (e.g., “communication,” “joint action,” etc.) best predict brain response patterns.

⸻

2. Feature Extraction via CLIP
	1.	Video Embeddings:
	•	Split each video into frames (or evenly sampled frames).
	•	Use CLIP’s image encoder to get embeddings for each frame.
	•	Aggregate frame embeddings (e.g., average pooling) → yields 1 vector per video.
- This is your “video embedding”.
	2.	Dimension-Specific Scores:
	•	For each dimension (e.g., “communication,” “dominance,” “cooperation”), create a short descriptive prompt (e.g., “a picture of people actively communicating”).
	•	Obtain text embeddings for these prompts via CLIP’s text encoder.
	•	Compute cosine similarity between the text embedding and each video embedding → yields a single dimension score per video.
- This produces, for each dimension, a list of 244 scores (one per video).

⸻

3. Video-by-Video Similarity Matrices
	•	For each dimension (e.g., “communication”), you have 244 scalar scores (one score per video).
	•	Construct a video-by-video similarity matrix for that dimension:
- e.g., similarity(video_i, video_j) = |score(i) – score(j)| or use correlation of sets of dimension scores.
- Another approach: Build a similarity matrix directly from the CLIP text–video embeddings (though you’ve effectively done that via “dimension score”).

⸻

4. Neural Similarity & Representational Similarity Analysis (RSA)
	1.	Neural Similarity:
	•	For each pair of videos (video_i, video_j), compute similarity in brain response patterns (e.g., correlation across voxels in an ROI).
	•	Yields a neural RDM (Representational Dissimilarity Matrix) or neural similarity matrix for the 244 videos.
	2.	Comparing Neural to Semantic Matrices:
	•	For each dimension’s similarity matrix (semantic RDM), compare it to the neural RDM via Spearman’s correlation.
	•	The dimension that obtains the highest correlation with neural data presumably best explains that region’s representation.

⸻

5. Extensions & Variations
	•	Multiple ROIs: Repeat the RSA in different brain regions or across the entire cortical surface (encoding models).
	•	Dimension Library: You can expand your “semantic dimension” prompts beyond “communication,” e.g., “domination,” “prosocial,” “emotionally arousing,” “object-directed,” etc.
	•	Fine-tuning: If needed, adapt CLIP to your social-video domain (though zero-shot may suffice).
	•	Cross-Subject Consistency: Compare how consistent dimension-driven neural patterns are across participants.

⸻

Summary

Your approach uses CLIP to generate video embeddings and obtains dimension-specific scores by comparing video embeddings to dimension text prompts. You then construct similarity matrices for each dimension and correlate these with neural similarity matrices in an RSA framework. This pipeline should highlight which semantic dimension (communication, cooperation, etc.) best aligns with the neural responses to social interactions.
Through this apporach, we can use a data-driven way to decode the interpretable dimension.


**The core here is actually based on extracting scores of videos in specific dimensions using CLIP, and then associating them with neurons, thereby understanding the neural activity in interpretable dimensions.**

Setting a well-defined search range is invaluable because it focuses exploration on a clearly bounded set of dimensions. This not only increases efficiency (you’re not chasing irrelevant features) but also supports a cohesive theoretical framework: if you can methodically sweep through a dimensional space (e.g., social primitives vs. emotional attributes), you can confidently say, “Neural signals show no difference along Dimension X, yet strong differences along Dimension Y.” That in turn builds a more principled account of which dimensions truly matter for neural coding of social interactions.


videos as bridge and anchor

**how to define the range (design space)?**
**consider full grid space first**
describe landscape: x y z 

dimensional: continuous

plan1: 给定grid, 计算 
    anchor word: define polar (for interpretability)
    dense space (take a part of the embedding space) image is also there
plan2: theretical dimensions
plan3: word list

# active learning




memorability



---

now we have the pipeline: dimension (word) --> score (correlation)
    word -LLM prompting-> -CLIP encode-> CLIP embedding -cosine similarity-> dimensional annotation for video --> RSA (with neural) -rank correlation-> dimensional representational likelihood
    from a given dimension (a word), we can have the likelihood of neural activity encoding it.
with validations

dimension (word)
    single dimension
    multiple dimension (hierarchical structure)

then it becomes an optimization problem: find the dimension that yields highest likelihood, and that dimension is the one neural activity is encoding

the problem now is, what is the design space? **what are the potential dimensions we would like to examine?**
    if this space is large, we migh consider using the active learning loop to explore more efficiently. But let's start from the conceptual level, what would be the space?

few ways to define the design space:
1. theory-driven: synthesize previous literatures
    formalize: a concept into description
    **hierarchical organization: relationship = siblings + friends + couple ....**
        multiple dimensions: softmax: clasification task
        contrast

2. automatic semi-driven discovery: maybe find some unexpected dimension, like hand (isc 2004 science)
obtained from embedding space, like theory-driven, but without that kind of strong prior, using active learning to discover dimension

there are several **criterions** for this design space:
1. **videos are informative** --> the dimensions should have relatively wide distribution among videos 
2. taxtonomy: we are looking for dimensions, not instance? 
3. No. *shall we limit this to social interaction? relevant dimensions/irrelevant*

3. data-driven:
clustering the videos, find the potential dimensions (fishing....)
dimension reduction
clustering



after we have the design space, we can go through it, to build a landscape
then we can find the optimal, as well as the overall pattern (as the brain might be encoding multiple dimensions simutaneously)



# landscape
embedding: which dimensions --> dimensions
hierarchical concept: multiple labels: classification problem --> dimension label, dimension score
dimensional representation
    relationship related information
    theory-driven
        literature
        embedding space (GPT)

control group:
    single label/item
    orgnaize into hierarchical structure: wordNet
    data-driven: 
    ensure (threshold: distribution)




dimension-pipeline



I abandon the idea of 
embedding space and active learning
 active learning is just an efficient sampling method, the design space matters; it requires continuous space
 embedding space: too broad, the landscape is not interpretable, especially when it is continuous
    anchor?




order









检查代码是不是写错了
显著性检验

关注少量核心区域
    而不是全脑
    可以验证
    全脑的寻找

验证型的分析
新的发现






--
# plan

visual vs. CLIP
    dimensional

1. 从词到维度的annoatation
    video embedding
        stimuli to embedding
    dimensional annotation
        LLM-CLIP (theory to embedding)
            **modeling stimuli**
        single
        multiple (hierarchical), polar [TODO] 
            validate [TODO]

2. 从文献中找到一些维度
    [TODO]
    embedding space: other dimensions? 
标准化

3. dimensions, design space
遍历 看和神经的相关
    检查：数据对齐(video index) [TODO]
    显著性检验 [TODO]


4. 结果解读
    Which RoI is encoding ....
        known, validate
        unknown, discovery

visual

5. even more: active learning in embedding space





Question: Which dimension is represented by brain?
Model (semantic) representation vs. neural  representation

0. video preparation (matching)
1. video embedding vector
    videos
        validate: 1) within vs. between 2) dimension reduction
    video_embedding_df
2. conditional annotation of video 
    concept, video_embedding_df
        dimension_list; multiple: softmax
    video_annotation_df
3. neural data
    check, organize
    neural activity -mask -summarize by roi
4. construct rsm:
    df
    rsm
        multiple: organize
    rsms (previous: neural, rating, CLIP overall, visual)
5. compare video annotation with neural
    word, neural level
        word-video_annotation-video_annotation_df-video_annotation_rsm
        correlate with neural_rsm
    correlation coefficient (optimization goal)
6. go through design space
    define design space
        discrete (from theory, LLM)
        continuous embedding space
    for each word, run this 
        landscape
        summit

potential results
Region decodes ....



TODO
1. write functions 
    preparation
    word to correlation
2. read papers
    design space (concept list)
    
