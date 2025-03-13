# neural
## ROI
ROI 我使用了文章中的mask，对GLM beta进行masking。
文章中的ROI mask通过 Anatomical ROIs (EVC and MT)， Anatomically constrained functional ROIS (pSTS-SI and aSTS-SI)， Functional ROIs (FFA, STS-Face, PPA, EBA, and LOC)。

averaged across runs


我 对于单独的ROI比较voxel
RoI-feature
correlation (permutation matrix)
regression (test?)



获取RSM函数

遍历
    group avg + sub

提取指定区域的xyz
变成df

遍历 组织



neural df整理
neural rsm整理
npy



回归：
    rank 神经
    其他不rank

    GLM

    stats: 模型输出
RSMs were predicted by the distance in each of the three properties

pitfall:
    video ordering
    rank
    distance





 ROI+ whole brain searchlight
        如何对齐 呈现: freesurfer
        **未来展望**



neural:
T1W



data_for_rsm
nested dict: df

df->rsm
nested dict


rename
pairwise





neural √
    -subj --> group
        -roi
            -side
    (video x voxel)
semantic √
    - dimension1 (relation)
    - dimension2 (context)
    - overall similarity ?
    ...
    (video x sub_dim)
NN  [TODO]
    - CLIP
    - ResNet 1
    (video x embedding)
rating
    - overall
        - dimension
    (video x dimension)


RSA [TODO]
1. 
go through the nested dict --> rsm dict √

2. 
each two rsms --> correlation with permutation p √

3. 
two dicts: go through-->pairwise r and p
dict1 rsm1 dict2 rsm2 r p √

4. all the dict
dict1 rsm1 dict2 rsm2 r p √
顺序 优先级



转换所有df

resnet

模型比较

subj关联 group合并
计算noise ceiling







回归
    就是这个数据结构
    subj x feature
    我如何进行定量的统计检验呢
    得到某个feature的回归系数在group level是否显著


找维度



---
finalize
结果


future: searchlight, different layers
two papers


visualization; title
显示显著



---
space
[
    {
        "dimension": "body_orientation",
        "description": "How individuals position or face their bodies relative to each other, potentially signaling interaction (e.g., face-to-face) or disengagement.",
        "examples": ["direct facing vs. angled postures", "body turned away but head oriented toward the other"]
    },
    {
        "dimension": "perceptual_binding",
        "description": "The extent to which two bodies are visually processed as a cohesive 'unit' rather than two independent figures, supporting early recognition of a social encounter.",
        "examples": ["holistic perception of a pair sharing space", "integrated figure-ground grouping of two people"]
    },
    {
        "dimension": "mutual_reachability",
        "description": "Whether individuals appear poised to physically or communicatively engage each other, indicating potential for interaction.",
        "examples": ["arms moving toward a shared point of contact", "leaning in to speak or gesture"]
    },
    {
        "dimension": "dyadic_emphasis",
        "description": "How strongly the scene highlights two specific agents (rather than a crowd), enabling quick detection of person-to-person social events.",
        "examples": ["two-person grouping isolated from background", "clear postural synergy contrasting with other elements"]
    },
    {
        "dimension": "action_predictability",
        "description": "The degree to which a next action (e.g., handshake, greeting, or confrontation) can be anticipated from body arrangement or stance.",
        "examples": ["opened arms hinting at a hug", "hands raised in a defensive or combative posture"]
    }
]
G 2024


[
    {
        "dimension": "relationship",
        "description": "Indicates the broader social tie or affiliation between the two individuals (for example, hierarchical vs. egalitarian, close vs. distant).",
        "examples": ["power dynamics", "friendly rapport", "co-worker bonds"]
    },
    {
        "dimension": "interaction_intent",
        "description": "Describes why the individuals might be engaging — for instance, the underlying purpose or motive in the interaction.",
        "examples": ["mutual cooperation", "explicit communication", "persuasion attempts"]
    },
    {
        "dimension": "emotional_climate",
        "description": "Reflects the general affective tone or emotional backdrop, independent of specific discrete emotions.",
        "examples": ["tension vs. ease", "enthusiasm vs. indifference", "warmth vs. coldness"]
    },
    {
        "dimension": "shared_goal_alignment",
        "description": "Captures whether the individuals’ goals are in conflict or harmony, beyond mere cooperation details.",
        "examples": ["fully aligned objectives", "competitive stances", "neutral parallel pursuits"]
    },
    {
        "dimension": "social_primitives",
        "description": "High-level cues related to spatial or bodily configuration indicative of potential interaction states.",
        "examples": ["physical proximity", "extent of facing or orientation", "joint attention"]
    },
    {
        "dimension": "communicative_mode",
        "description": "Represents the nature of how information is exchanged (verbal or nonverbal) without specifying exact behaviors.",
        "examples": ["predominantly nonverbal exchange", "spoken hints with minimal gestures", "multimodal cues"]
    },
    {
        "dimension": "dominance_vs_submission",
        "description": "Highlights whether one individual appears to be leading or controlling and the other following or yielding.",
        "examples": ["one person taking charge", "balanced peer interaction", "one deferring consistently"]
    },
    {
        "dimension": "affective_arousal",
        "description": "Captures the level of emotional activation within the interaction, irrespective of positivity or negativity.",
        "examples": ["calm or subdued energy", "high excitement or tension", "mild vs. intense activation"]
    }
]
review NRP


Here is a structured list of social interaction dimensions extracted from the provided documents, formatted in JSON:

[
    {
        "dimension": "physical proximity",
        "description": "The spatial distance between interacting individuals.",
        "examples": ["Close physical interaction", "Distant interaction", "Remote communication"]
    },
    {
        "dimension": "facingness",
        "description": "The extent to which individuals are oriented toward each other.",
        "examples": ["Face-to-face interaction", "Side-by-side interaction", "Back-to-back interaction"]
    },
    {
        "dimension": "joint action",
        "description": "The degree of coordination between individuals in achieving a shared goal.",
        "examples": ["Collaborative problem-solving", "Cooperative work", "Synchronized movements"]
    },
    {
        "dimension": "communication modality",
        "description": "The channel through which individuals exchange information.",
        "examples": ["Verbal communication", "Nonverbal gestures", "Written messages"]
    },
    {
        "dimension": "emotional valence",
        "description": "The affective tone of the interaction, ranging from negative to positive.",
        "examples": ["Friendly conversation", "Hostile argument", "Neutral exchange"]
    },
    {
        "dimension": "arousal level",
        "description": "The intensity of emotional engagement in the interaction.",
        "examples": ["Excited celebration", "Calm discussion", "Tense confrontation"]
    },
    {
        "dimension": "power dynamics",
        "description": "The balance of authority and influence between individuals.",
        "examples": ["Leader-follower dynamic", "Equal-status peers", "Subordinate-superior interaction"]
    },
    {
        "dimension": "social role",
        "description": "The functional roles that individuals assume in an interaction.",
        "examples": ["Mentor-student", "Parent-child", "Teammates"]
    },
    {
        "dimension": "intentionality",
        "description": "The degree to which actions are planned versus spontaneous.",
        "examples": ["Deliberate persuasion", "Spontaneous help", "Unintentional interference"]
    },
    {
        "dimension": "social norms",
        "description": "The extent to which the interaction adheres to cultural and contextual expectations.",
        "examples": ["Formal business meeting", "Casual social gathering", "Ritualized greetings"]
    },
    {
        "dimension": "agency distribution",
        "description": "The extent to which individuals have autonomy in the interaction.",
        "examples": ["Mutual decision-making", "One-sided command", "Negotiation"]
    }
]

nhb


[
    {
        "dimension": "Physical Proximity",
        "description": "The spatial distance between interacting agents, which can influence the perception and nature of the interaction.",
        "examples": ["Close physical contact", "Distant interactions", "Crowded environments"]
    },
    {
        "dimension": "Facingness",
        "description": "The extent to which agents are oriented toward each other, affecting engagement and interaction recognition.",
        "examples": ["Face-to-face interaction", "Side-by-side interaction", "Back-to-back positioning"]
    },
    {
        "dimension": "Contingent Motion",
        "description": "The synchronization or temporal coordination of movement between agents.",
        "examples": ["Dancing together", "Mimicking gestures", "Simultaneous reaching for an object"]
    },
    {
        "dimension": "Goal Compatibility",
        "description": "The extent to which agents share or oppose goals during the interaction.",
        "examples": ["Cooperative tasks", "Competitive games", "Helping vs. hindering behavior"]
    },
    {
        "dimension": "Communicative Intent",
        "description": "The degree to which communication, whether verbal or non-verbal, is present in an interaction.",
        "examples": ["Speaking and gesturing", "Silent coordination", "Signaling intentions"]
    },
    {
        "dimension": "Affect and Valence",
        "description": "The emotional tone of the interaction, including whether it is positive or negative.",
        "examples": ["Friendly conversation", "Aggressive confrontation", "Neutral exchange"]
    },
    {
        "dimension": "Social Role Differentiation",
        "description": "The distinction between roles taken by individuals in a social interaction.",
        "examples": ["Leader-follower dynamic", "Teacher-student relationship", "Service provider-client interaction"]
    },
    {
        "dimension": "Hierarchy and Power Dynamics",
        "description": "The relative status or authority differences between interacting agents.",
        "examples": ["Boss-employee discussion", "Peer-to-peer collaboration", "Parent-child interaction"]
    },
    {
        "dimension": "Social Relationship Context",
        "description": "The nature of the relationship between agents, shaping expectations and interaction styles.",
        "examples": ["Stranger interactions", "Familial bonds", "Close friendships"]
    },
    {
        "dimension": "Interaction Formality",
        "description": "The structuredness or casualness of an interaction, affecting conventions and expected behaviors.",
        "examples": ["Business meeting", "Casual small talk", "Ceremonial interactions"]
    }
]
Qua 2017


[
    {
        "dimension": "social primitives",
        "description": "Basic visual cues indicating the presence of a social interaction.",
        "examples": ["agent distance", "facingness", "contingent motion"]
    },
    {
        "dimension": "physical interaction",
        "description": "The extent to which individuals physically engage with each other.",
        "examples": ["pushing", "handshaking", "hugging"]
    },
    {
        "dimension": "communicative interaction",
        "description": "The presence of non-physical but socially meaningful exchanges.",
        "examples": ["gesturing", "talking", "signaling"]
    },
    {
        "dimension": "goal compatibility",
        "description": "The alignment or opposition of intentions in an interaction.",
        "examples": ["cooperation", "competition", "helping vs. hindering"]
    },
    {
        "dimension": "valence",
        "description": "The emotional tone of the interaction, indicating its positivity or negativity.",
        "examples": ["friendly", "hostile", "neutral"]
    },
    {
        "dimension": "social roles",
        "description": "The roles individuals assume within an interaction.",
        "examples": ["leader vs. follower", "aggressor vs. victim", "teacher vs. student"]
    },
    {
        "dimension": "group structure",
        "description": "The number and configuration of individuals involved in an interaction.",
        "examples": ["dyadic interaction", "small group", "crowd"]
    }
]


自己想

summary
1. 合并重复 变得工整
2. 选项: unipolar, ordered, classification

design space


汇总
调用函数
获得conditional embedding dict


这边运行pairwise
储存


















ViT 是 由 Google 在 2020 年提出 的 基于 Transformer 的图像分类模型，不同于传统 CNN（如 ResNet），它使用 自注意力机制（Self-Attention） 来建模图像特征。




写一个函数compute_sequence_similarity
input: sequence1 sequence2 similarity_metric
output: similarity_value
具体来说
首先删掉na的位置


根据similarity_metric=
cosine similarity
euclidean distance (使用1-distance)
jaccard




写一个函数compute_rsm
input: df，第一列是video_name; similarity_metric; video_name_list; if_display
行是video_name 列是features
output: pairwise similarity matrix
每一个视频可以描述为feature向量

首先根据video_name_list筛选并排序

计算两两视频之间的feature相似性
output: pairwise similarity matrix

如果if_display
展示该热图，用virdis colormap
不用video_name而使用video_name在video_name_list中的index作为坐标label


写一个函数correlate_rsm
input: rsm1, rsm2, n_permutation, if_display
output: r, p
提取各个rsm的下三角 计算rank correlation

并且执行多次permutation:
随机打乱第一个matrix的行，然后计算相关
得到null distribution (次数为n_permutation)
然后observed similarity在这个null distribution的位置
设置one-tail p的数值为permutation_p
onetail公式： p = (1 + number of null r >=  observed r) / (1 + number of permutations)

如果if_display=True
则展示null distribution直方图
并且用垂直线显示observed r
annotation r= p=




