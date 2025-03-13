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

0. video preparation (matching)√ [match_videos.py]
1. video embedding vector√ [extract_video_embedding_CLIP.ipynb]
    videos
    embedding_df: CLIP_video 
         validate: within vs. between √ 
         description: 1) distribution √ 2) clustering √ [clustering_CLIP_embedding.ipynb]
2. neural data
    check, organize
    neural activity -mask -summarize by roi
        which level
    more RoI? **whole-brain**
    specific RoI

3. construct rsm:
    df
    rsm
        multiple: organize
    rsms (previous: neural, rating, CLIP overall, visual)

4. RSA: neural, CLIP, ResNet
    CLIP is better than ResNet
    not satisfying, multiple dimensions--need clear separation

    statistical test

4. conditional annotation of video (dimensional)
    concept, video_embedding_df
        dimension_list; multiple: softmax
    video_annotation_df
    validate:
        example | distribution, extreme
        human rating correlation

5. compare video annotation with neural
    word, neural level
        word-video_annotation-video_annotation_df-video_annotation_rsm
        correlate with neural_rsm
    correlation coefficient (optimization goal)

6. go through design space
    define design space
        **discrete (from theory, LLM)**
        continuous embedding space, active learning
    for each word, run this 
        landscape
        summit

potential results
Region decodes .... (validate & discovery)



TODO
1. write functions 
    preparation
    word to correlation
2. read papers
    design space (concept list)
    
mapping


frame embedding --> video embedding --> video unipolar annotation (cos) --> video structure (bi- or multi- polar) annotation(classification standarize)-->video dimension vector


neural RSM - semantic dimension RSM
which dimension is this area representing?
optimization problem









design space
how to choose dimension?
    active learning in embedding space:
        constrain: high-level (discrete), distance to social interaction (boundary)
    landscape



network的形状
semantic的map

在其中游走
发现新的


frame embedding √ --> video embedding √--> video unipolar annotation (cos) √ --> video structure (bi- or multi- polar) annotation ƒ(classification standarize)-->video dimension vector √ (df) |multiple

hierarchical semantic structure (dict)
-->
all the semantic df√ [dimensional_CLIP_annotation.ipynb]



neural df
    check
    searchlight RSM [TODO]

1. 使用对齐的 --> TPJ STS EVC
2. 写基础的RSM
3. searchlight RSM


函数建构RSM


函数for annotation RSM

对比
显著性检验

新的维度：文献 综合

对比的文献
理论文献

---
RSM [TODO]
    df -> RSM
    semantic    
    neural RSM
test

RSA: neural (RoI) --> neural RSM [TODO]

neural RSM - semantic dimension RSM
which dimension is this area representing?
optimization problem


design space
how to choose dimension?
    active learning in embedding space:
        constrain: high-level (discrete), distance to social interaction (boundary)
    landscape

理论




how:
achtype: dimensional
where:
embedding hierarchical clustering....

简单的维度 (聚类)

文章

LLM




请列出所有与 "social interaction" 相关的概念，并根据其层次关系归类。
例如：
- 关系 (relationship): 亲子关系、朋友关系、权力关系
- 情感 (emotion): 亲密、信任、害怕、愤怒
- 交流 (communication): 语言交流、非语言交流、书面交流
- 场景 (scene): 工作场合、家庭环境、公共场所

扩展?





3 更多CLIP 50
    编号
    dim
rating
    self: 对比DL
    neural: DNN强大
    整理-回归

2 自动发掘 基于embedding
    调研
    合并

    设计子维度

    检查
    根据相关情况
    合并

    找到最佳维度

    解释: 相关

重要




新的CLIP
rating
几个维度 直接操作

写作
    先交一个link




---

写作

先比CLIP和其他 找到优势模型
再来可解释性
open-window
可解释性 特异性




写一个函数
input: 
1. roi (roi_side的形式)
2. candidate_list (字典，表明了每个module里选哪些rsm)
3. candidate_rsm_dict (dict_dict 比如combined_rsm)
4. neural_rsm

从neural_rsm里提取subject在这个roi_side的rsm
再从candidate_rsm_dict中提取相应的

然后提取下三角

神经的做rank处理，特征的不处理

合并为df

sub video1 video2 roi feature1 feature2 ...

返回df


第二个函数
input：上面那个df
基于刚才这个df 建立回归模型
打印不同feature的回归系数
如果if_display
则展示不同feature的coefficient，从大到小(不包括intercept)
返回dict {roi: {intercept: intercept, feature1: beta1, feature2: beta2}}


    lmm 检查
    颜色  选择维度 颜色
    整合绘图

    consistency: rank; cross data
    mixed model
    bonferroni

    提mixed effect
    可视化




第二个函数
input: average_df 根据这里面的r 挑选那些r在0.2以上的roi-candidate组合
对于每一个roi, 都选择其所有的candidate
然后调用上面那个函数获得其回归系数
然后汇总所有roi的
返回dict
    汇总的显示形式
    统一的一组
    CLIP annotation + resnet50

