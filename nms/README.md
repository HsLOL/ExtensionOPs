### HBB_NMS_CPU  
1. you can use`test_HBB_nms_cpu.py`to test HBB_NMS_CPU extension, but you should reference HBB_NMS_CPU/README.md to create conda environment, and install `opencv-python` to show image.  
2. Note: the **test_HBB_NMS_CPU_after.png** and **test_HBB_NMS_CPU_before.png** is the result.  
### HBB_Soft_NMS_CPU  
1. you can use `test_HBB_soft_nms_cpu.py` to test HBB_Soft_NMS_CPU extension, but you should reference HBB_Soft_NMS_CPU/README.md to create conda environment, and install `opencv-python` to show image.  
2. Note: the **test_HBB_Soft_NMS_CPU_after.png** and **test_HBB_Soft_NMS_CPU_before.png** is the result.  
### HBB_NMS_GPU
1. you can use `test_HBB_nms_gpu.py` to test HBB_NMS_GPU extension, but you should reference HBB_NMS_GPU/README.md to create conda environment, and install `opencv-python` to show image.
2. Note: the **test_HBB_NMS_GPU_after.png** and **test_HBB_NMS_GPU_before.png** is the result.
### OBB_NMS_GPU
1. you can use `test_OBB_nms_gpu.py` to test OBB_NMS_GPU extension, but you should reference OBB_NMS_GPU/README.md to create conda environment, and install `opencv-python` to show image.
2. Note:  
   - the **test_OBB_NMS_GPU_after.png** and **test_OBB_NMS_GPU_before.png** is the result.  
   - variable boxes in `test_OBB_nms_gpu.py` must be on device cuda:0, otherwise will occur error!
