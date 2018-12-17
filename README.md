# multiple-level-fearture-BIQA
This is the reproduced work from ICIP 2018's paper "MULTIPLE LEVEL FEATURE-BASED UNIVERSAL BLIND IMAGE QUALITY ASSESSMENT MODEL"

## TODO List:


## principle 
The net in this paper is combine the resnet and a finetune block, grab the 4 blocks feature map as the encoder input, the encoder combine with 1x1 conv, 3x3 conv and a GAP

## restore the network
   type: resnet_v1_50
   
## finetune block design
   1x1 conv
   3x3 conv
   Gap
  
## dataset generation
   train [8]:[2] testing in Live IQA

## recent result

### time 2018.12.17

| epochs | SRCC | PLCC |
| :----: | :---: | :---: |
| 1      | 0.9342575998390334 | 0.9262977640507852 |
| 2      | 0.9545877222378376 | 0.9618316040597912 |
| 3      | 0.9530796903070575 | 0.9634742645198329 |
| 10      | 0.9604244665165107 | 0.9680835927122491 |
| 11      | 0.9615523664509095 | 0.9697664157606344 |
| 12      | 0.9603307561826612 | 0.9681378304542274 |
| 13      | 0.9599215513282451 | 0.9681977447704329 |
| 20      | 0.9613756915051567 | 0.9684274591303199 |
| 21      | 0.9655127342977855 | 0.969946051186061 |
| 22      | 0.9619355103706009 | 0.969665080120685 |
| 23      | 0.962235117159276 | 0.9692066745224719 |




