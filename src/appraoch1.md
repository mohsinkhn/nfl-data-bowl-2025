Based on analysis of data it seems that, the outputs are smooth curves which seem continuation of pre-throw curves. They also seem impacted by the direction and location of where ball which land in addition to their own speed and orientation.

Given this, I would for us to setup a seq2seq model. Given very high corrlation in input either RNN/Convolution network might be suitable for both encoder and decoder. For decoder, we have been decoder length. 

We can write a code in such that, teaching forcing and different archs for encoder and decoder are supported.

Data loading and normalization would be critical here. To begin with, we can just use a fixed encoder length with padding. A quick analysis of pre-throw frames would help us decide this. We can probably pick 90th percentile num of frames.

For each player we can create new sample. Essentially making (game_id, play_id, nfl_id) unique for each sample. We can choose normalisation such that last pre-throw point is at (0, 0) (We can later add this offset to predictions). Also, we can have an option to normalize data points (x, y, orientation) such that , players are going from left to right. In addition, we can normalize (x,y) by ground height, width.  