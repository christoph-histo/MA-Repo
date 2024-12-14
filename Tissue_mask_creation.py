import os
import slicer
import nibabel as nib
import numpy as np
import json
from scipy.ndimage import zoom


def process_files_in_directory(base_directory):
    for root, dirs, files in os.walk(base_directory):
        # Filter directories that end with "_tissue_segmentation"
        tissue_dirs = [d for d in dirs if d.endswith("_tissue_segmentation")]

        for tissue_dir in tissue_dirs:
            tissue_dir_path = os.path.join(root, tissue_dir)
            print(f"Processing directory: {tissue_dir_path}")

            # Filter for files ending with "tissue.stl"
            stl_files = [f for f in os.listdir(tissue_dir_path) if f.endswith("tissue.stl")]
            raw_files = [f for f in os.listdir(tissue_dir_path) if f.endswith(".raw")]

            for raw_file in raw_files:
                raw_path = os.path.join(tissue_dir_path, raw_file)

                # Match .stl files to the corresponding .raw files based on prefix
                raw_base_name = os.path.splitext(raw_file)[0]
                for stl_file in stl_files:
                    stl_base_name = "_".join(stl_file.split("_")[1:-1])  # Extract common prefix
                    if stl_base_name in raw_base_name:
                        stl_path = os.path.join(tissue_dir_path, stl_file)
                        nhdr_path = raw_path.replace(".raw", ".nhdr")

                        create_np_vols(nhdr_path, stl_path)



def create_np_vols(reference_volume_path,stl_file_tissue):
    referenceVolumeNode = slicer.util.loadVolume(reference_volume_path)
    segmentationNodeTissue = slicer.util.loadSegmentation(stl_file_tissue)

    tissue_name, extension = os.path.splitext(os.path.basename(stl_file_tissue))
    segmentIdTissue = segmentationNodeTissue.GetSegmentation().GetSegmentIdBySegmentName(tissue_name)
    
    seg_mask_tissue = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNodeTissue,segmentIdTissue,referenceVolumeNode)
    seg_mask_tissue = seg_mask_tissue.transpose((2, 1, 0))

    np.save()



if __name__ == "__main__":
    base_directory = "/home/histo/Dokumente/christoph/Masterarbeit/Datensätze"  
    process_files_in_directory(base_directory)

