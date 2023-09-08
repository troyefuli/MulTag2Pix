# MulTag2Pix: Tag-adaptive Multi-character Line Art Colorization with Text Tag
[Abstract]   Line art illustrations often depict multiple characters, yet existing automatic colorization approaches typically focus on single-character line art, leaving a gap in research for multi-character line art colorization. Moreover, automatic colorization methods without user guidance often fail to meet users' personalized needs. To address these issues, we propose MulTag2Pix, a multi-character line art colorization network based on color text tags, which caters to personalized and interactive colorization demands. In addition, existing approaches often fail to adaptively color and coordinate with user-provided color text tags, leading to color errors. Then we innovatively give a Tag-Adaptive Colorization Refinement module (TACR) to handle the challenges. Furthermore, the color bleeding issues severely affect the quality of colorization. Therefore, a dual-branch encoder embedded with skeleton maps is utilized to fuse different sources of information, providing more accurate region segmentation and avoiding inconsistent colors within a given semantic region. Experimental results on a large-scale illustration dataset show that our network achieves high-quality personalized colorization of line art across multiple characters. Furthermore, our network also outperforms state-of-the-art methods in single-character line art colorization.

## dataset
The link of the dataset is as follows, currently only the test set is placed. 

    [dataset] (Link：https://pan.baidu.com/s/1tZdqZbXQUSGyhqHyd3Ptpw?pwd=bjgb  Extraction code：bjgb)

## An example of our method
![image](https://github.com/troyefuli/MulTag2Pix/blob/main/img/show.jpg)

