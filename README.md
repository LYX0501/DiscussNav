# DiscussNav

In this work, we introduce a novel zero-shot VLN framework. Within this framework, large models possessing distinct abilities are served as domain experts. Our proposed navigation agent, namely DiscussNav, can actively discuss with these experts to collect essential information before moving at every step. These discussions cover critical navigation subtasks like instruction understanding, environment perception, and completion estimation. The performances on the representative VLN task R2R show that our method surpasses the leading zero-shot VLN model by a large margin on all metrics.

![DiscussNav](https://github.com/LYX0501/DiscussNav/blob/main/DiscussNav.gif)

## Requirements
Ubuntu 18.04.6 LTS

Python 3.8.17

Torch 1.13.1

[Matterport3DSimulator](https://github.com/xinyu1205/recognize-anything)

[Recognize Anything](https://github.com/xinyu1205/recognize-anything) (RAM)

[InsturctBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)

