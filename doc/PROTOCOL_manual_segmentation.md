# Protocol for manual segmentation of left atrial appendage w/ 3D Slicer and monailabel AIAA


# NOTABLES
6184188635_0000.nii.gz - filling defect 
2549981393_0000.nii.gz - filling defect 
1219145927_0000.nii.gz - filling defect 


# monailabel server 
# connect to RHEL server 
ssh vac10mulspd219.spd.vaec.va.gov
# connect to docker container/start monailabel server 
docker ps -a 
docker start [container_id] (if stopped)
docker exec -it [container_id] /bin/bash 
monailabel start_server --app apps/radiology --studies /opt/monai/datasets --conf models segmentation

# 3D Slicer 
MONAI Label Server: http://10.249.63.89:8000
# shortcuts 
-Ctrl scroll to zoom
-Ctrl scroll to adjust paint brush diameter
-shift to sync 3D slices at pointer location
-right click, Window/level presets -> CT Bone (Can adjust W/L for CT Bone in Volumes)
# initial labels
1. Add new label in Segment Editor [Segment_1]
2. Select Paint, Sphere brush 
3. Perform dilated segmentation of LAA using Segment_1 label
    -include all of LAA
    -surrounding dark areas are okay
    -exclude/erase bright areas that are not LAA
4. Review/Edit in remaining views (e.g., sag, trans)
5. Remove "Sphere brush" if necessary for final edits 
6. Switch label to LAA 
7. Select Threshold from Segment Editor
8. In "Masking" change Editable Area to "Inside Segment_1"
9. Zoom in on slice with variable intensity in LAA 
10. Adjust bottom threshold level to appropriate level
11. Select "Apply"
12. Remove Segment_1 visibility 
13. Scroll through and make any final corrections 
# AIAA
1. Select Paint, Sphere brush
2. Scroll through each view adjust segmentation 
    -include all of LAA
    -surrounding dark areas are okay
    -exclude/erase bright areas that are not LAA
3. Review/Edit in remaining views (e.g., sag, trans)
4. Select Threshold from Segment Editor
5. In "Masking" change Editable Area to "Inside laa"
6. Zoom in on slice with variable intensity in LAA 
7. Adjust bottom threshold level to appropriate level
8. Select "Apply"
9. Scroll through and make any final corrections 


