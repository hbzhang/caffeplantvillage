 #!/bin/bash
 TEST_FOLDER_NAME="/hdd/plantvillage/AlexNet/crowdai/test"
  for file in `ls $TEST_FOLDER_NAME`
  do
      echo $file
      convert $TEST_FOLDER_NAME/$file -resize 256x256! $TEST_FOLDER_NAME/$file
  done

 
	   
