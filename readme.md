# Florence-2 Lane Detection 

Florence-2 is a powerful yet relatively small multimodal model that can perform a variety of tasks. This repository contains code for fine-tuning Florence-2 model to do lane detection using TuSimple dataset. This is a simple demonstration of how to use Florence-2 for tasks that are not in the list of supported tasks 
- Captioning
- Optical Character Recognition
- Object Detection (2D bounding box with class label)
- Region Detection 
- Region Segmentation
- Vocabulary Segmentation

I have not tried the other tasks, but I believe it should be similar to the lane detection task that contains text output and/or point sets output, such as 2D pose estimation.

As shown below, the results and generalization are not perfect as the Tusimple_lane dataset is small. You can try yourself on larger dataset such as CurveLanes.

![Alt text](images/val_results/image_0_epoch_50.png)
![Alt text](imagesval_results/image_2_epoch_50.png)
![Alt text](images/val_results/image_4_epoch_50.png)
![Alt text](images/val_results/image_6_epoch_50.png)
![Alt text](images/val_results/image_8_epoch_50.png)

## Installation

```
conda env create -f environment.yml -n florence2
conda activate florence2
pip install -r requirements.txt
```



## Repository Structure

- `data_scripts/`: Scripts for data preprocessing and transformation
  - `tusimple_transform.py`: Processes the TuSimple dataset (https://www.kaggle.com/datasets/manideep1108/tusimple?resource=download) to fit the Florence-2 input format.

    Please combine the clips in train_set and test_set together, the  folder before transformation should be in the format below:
    
    ![Alt text](images/data.png)


    Then run 
    ```
    data_scripts/tusimple_transform.py --src_dir src_dataset_path --dst_dir dst_dataset_path --val
    ```

    to transform the data into the format of Florence-2, where (loc_x, loc_y) location in origianl image shape have been normalized to [0, 999] according to florence-2's input format.

    One line of 4 lanes in the annotaions.json after transforming is as follows:

    {"image": "tulane_0000.png", "prefix": "<OD_LANE>", "suffix": "lane<loc_493><loc_388><loc_488><loc_402><loc_482><loc_416><loc_475><loc_430><loc_469><loc_444><loc_464><loc_458><loc_457><loc_472><loc_451><loc_486><loc_445><loc_500><loc_439><loc_513><loc_433><loc_527><loc_427><loc_541><loc_421><loc_555><loc_415><loc_569><loc_409><loc_583><loc_403><loc_597><loc_396><loc_611><loc_391><loc_625><loc_385><loc_638><loc_378><loc_652><loc_372><loc_666><loc_366><loc_680><loc_360><loc_694><loc_354><loc_708><loc_348><loc_722><loc_342><loc_736><loc_336><loc_750><loc_330><loc_763><loc_324><loc_777><loc_317><loc_791><loc_312><loc_805><loc_306><loc_819><loc_300><loc_833><loc_293><loc_847><loc_288><loc_861><loc_282><loc_875><loc_275><loc_888><loc_269><loc_902><loc_264><loc_916><loc_257><loc_930><loc_251><loc_944><loc_245><loc_958><loc_239><loc_972><loc_233><loc_986>lane<loc_561><loc_388><loc_573><loc_402><loc_584><loc_416><loc_595><loc_430><loc_607><loc_444><loc_617><loc_458><loc_628><loc_472><loc_640><loc_486><loc_651><loc_500><loc_662><loc_513><loc_674><loc_527><loc_685><loc_541><loc_696><loc_555><loc_707><loc_569><loc_718><loc_583><loc_729><loc_597><loc_741><loc_611><loc_752><loc_625><loc_764><loc_638><loc_775><loc_652><loc_785><loc_666><loc_797><loc_680><loc_808><loc_694><loc_819><loc_708><loc_831><loc_722><loc_842><loc_736><loc_853><loc_750><loc_864><loc_763><loc_875><loc_777><loc_886><loc_791><loc_898><loc_805><loc_909><loc_819><loc_920><loc_833><loc_932><loc_847><loc_942><loc_861><loc_953><loc_875><loc_965><loc_888><loc_976><loc_902><loc_988><loc_916>lane<loc_415><loc_402><loc_392><loc_416><loc_370><loc_430><loc_347><loc_444><loc_325><loc_458><loc_302><loc_472><loc_279><loc_486><loc_257><loc_500><loc_234><loc_513><loc_211><loc_527><loc_188><loc_541><loc_165><loc_555><loc_142><loc_569><loc_120><loc_583><loc_97><loc_597><loc_75><loc_611><loc_52><loc_625><loc_29><loc_638><loc_7><loc_652>lane<loc_610><loc_375><loc_642><loc_388><loc_673><loc_402><loc_705><loc_416><loc_737><loc_430><loc_768><loc_444><loc_800><loc_458><loc_832><loc_472><loc_864><loc_486><loc_896><loc_500><loc_928><loc_513><loc_960><loc_527><loc_991><loc_541>"}

  - `curvelanes_transform.py`: Transform script for Curvelanes dataset,  available at https://github.com/SoulmateB/CurveLanes, dataset is too large (100k images) to train on a normal single machine, you can try yourself if you have a powerful workstation.

- `finetune.ipynb`: Contains the florence-2 fine-tuning code for lane detection

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TuSimple for providing the lane detection dataset
- Fine-tuning Florence-2 for VQA - https://github.com/andimarafioti/florence2-finetuning
- Florence-2: How it works and how to use it - https://www.assemblyai.com/blog/florence-2-how-it-works-how-to-use/

