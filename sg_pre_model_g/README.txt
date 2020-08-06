========================================================
Large-scale CelebFaces Attributes (CelebA) Dataset
========================================================

--------------------------------------------------------
By Multimedia Lab, The Chinese University of Hong Kong
--------------------------------------------------------

For more information about the dataset, visit the project website:

  http://personal.ie.cuhk.edu.hk/~lz013/projects/CelebA.html

If you use the dataset in a publication, please cite the paper below:

  @inproceedings{liu2015faceattributes,
 	author = {Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang},
 	title = {Deep Learning Face Attributes in the Wild},
 	booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
 	month = December,
 	year = {2015} 
  }

Please note that we do not own the copyrights to these images. Their use is RESTRICTED to non-commercial research and educational purposes.



========================
Change Log
========================

Version 1.0, released on 28/09/2015
Version 1.1, released on 23/03/2016, add landmarks annotations for align&cropped images
Version 1.2, released on 08/04/2016, add align&cropped images in lossless format
Version 1.3, released on 29/07/2016, add bounding box annotations for in-the-wild images
Version 2.0, released on 28/06/2017, add identity annotations note



========================
File Information
========================

- In-The-Wild Images (Img/img_celeba.7z)
    202,599 original web face images. See In-The-Wild Images section below for more info.

- Align&Cropped Images (Img/img_align_celeba.zip & Img/img_align_celeba_png.7z)
    202,599 align&cropped face images. See Align&Cropped Images section below for more info.

- Bounding Box Annotations (Anno/list_bbox_celeba.txt)
    bounding box labels. See BBOX LABELS section below for more info.

- Landmarks Annotations (Anno/list_landmarks_celeba.txt & Anno/list_landmarks_align_celeba.txt)
    5 landmark location labels. See LANDMARK LABELS section below for more info.

- Attributes Annotations (Anno/list_attr_celeba.txt)
	40 binary attribute labels. See ATTRIBUTE LABELS section below for more info.

- Identity Annotations (available upon request)
	10,177 identity labels. See IDENTITY LABELS section below for more info.

- Evaluation Partitions (Eval/list_eval_partition.txt)
	image ids for training, validation and testing set respectively. See EVALUATION PARTITIONS section below for more info.



=========================
In-The-Wild Images
=========================

------------ img_celeba.7z ------------

folder: img_celeba.7z.001, img_celeba.7z.002, ..., img_celeba.7z.014

---------------------------------------------------

Notes:
1. Please unzip these files together.

---------------------------------------------------



=========================
Align&Cropped Images
=========================

------------ img_align_celeba.zip ------------

format: JPG

------------ img_align_celeba_png.7z ------------

format: PNG
folder: img_align_celeba_png.7z.001, img_align_celeba_png.7z.002, ..., img_align_celeba_png.7z.016

---------------------------------------------------

Notes:
1. Images are first roughly aligned using similarity transformation according to the two eye locations;
2. Images are then resized to 218*178;
3. Please unzip "img_align_celeba_png.7z.*" together.

---------------------------------------------------



=========================
BBOX LABELS
=========================

------------ list_bbox_celeba.txt ------------

First Row: number of images
Second Row: entry names

Rest of the Rows: <image_id> <bbox_locations>

---------------------------------------------------

Notes:
1. The order of bbox labels accords with the order of entry names;
2. In bbox location, "x_1" and "y_1" represent the upper left point coordinate of bounding box, "width" and "height" represent the width and height of bounding box. Bounding box locations are listed in the order of [x_1, y_1, width, height].

---------------------------------------------------



=========================
LANDMARK LABELS
=========================

------------ list_landmarks_celeba.txt ------------

First Row: number of images
Second Row: landmark names

Rest of the Rows: <image_id> <landmark_locations>

------------ list_landmarks_align_celeba.txt ------------

First Row: number of images
Second Row: landmark names

Rest of the Rows: <image_id> <landmark_locations>

---------------------------------------------------

Notes:
1. The order of landmark locations accords with the order of landmark names;
2. The landmark locations in "list_landmarks_celeba.txt" are based on the coordinates of in-the-wild images;
3. The landmark locations in "list_landmarks_align_celeba.txt" are based on the coordinates of align&cropped images.

---------------------------------------------------



=========================
ATTRIBUTE LABELS
=========================

--------------- list_attr_celeba.txt --------------

First Row: number of images
Second Row: attribute names

Rest of the Rows: <image_id> <attribute_labels>

---------------------------------------------------

Notes:
1. The order of attribute labels accords with the order of attribute names;
2. In attribute labels, "1" represents positive while "-1" represents negative.

---------------------------------------------------



=========================
IDENTITY LABELS
=========================

---------------------------------------------------

Notes:
1. The face identities are released upon request for research purposes only. Please contact us for details;
2. There are no identity overlapping between CelebA dataset and LFW dataset.

---------------------------------------------------



=========================
EVALUATION PARTITIONS
=========================

------------- list_eval_partition.txt -------------

All Rows: <image_id> <evaluation_status>

---------------------------------------------------

Notes:
1. In evaluation status, "0" represents training image, "1" represents validation image, "2" represents testing image;
2. Identities of face images are NOT overlapped within this dataset partition;
3. In our ICCV 2015 paper, "LNets+ANet" is trained with in-the-wild images, while "FaceTracer" and "PANDA-l" are trained with align&cropped images;
4. Please refer to the paper "Deep Learning Face Attributes in the Wild" for more details.

---------------------------------------------------



=========================
Contact
=========================

Please contact Ziwei Liu (lz013@ie.cuhk.edu.hk) for questions about the dataset.