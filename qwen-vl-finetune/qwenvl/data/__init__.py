import re

available_corpus = dict(
    coco = ["/tos-bjml-video/linhan/datasets/llava_image_tune/coco/train2017"],
    gqa = ["/tos-bjml-video/linhan/datasets/llava_image_tune/gqa/images"],
    ocr_vqa = ["/tos-bjml-video/linhan/datasets/llava_image_tune/ocr_vqa/images"],
    vg = ["/tos-bjml-video/linhan/datasets/vg-dataset/images"],
    textvqa = ["/tos-bjml-video/linhan/datasets/llava_image_tune/textvqa/train_images"],
    activitynet = ["/tos-bjml-video/linhan/datasets/anet/ANet_320p_fps30/train"],
    # pope = ["phdd2:s3://coco-caption/val2014"],
    scienceqa = ["/tos-bjml-video/linhan/datasets/scienceqa/test"]
)

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "/fs-computility/video/shared/linhan/code/video_llava/Video-LLaVA/data_ft/ft_json/videochatgpt_tune_.json",
    "data_path": available_corpus["activitynet"][0],
}

LLAVA_IMAGE_TUNE = {
    "annotation_path": "/fs-computility/video/shared/linhan/code/video_llava/Video-LLaVA/data_ft/ft_json/llava_image_tune_.json",
    "data_path": "multi"
}
VIDEOCHATGPT_random6 = {
    "annotation_path": "/fs-computility/video/shared/linhan/code/video_llava/Video-LLaVA/data_ft/ft_json/random/random_0.06/random_video_0.06_single_video.json",
    "data_path": available_corpus["activitynet"][0],
}

LLAVA_IMAGE_TUNE_random6 = {
    "annotation_path": "/fs-computility/video/shared/linhan/code/video_llava/Video-LLaVA/data_ft/ft_json/random/random_image_0.06_single.json",
    "data_path": "multi"
}


LLAVA_IMAGE_TUNE_sel_groupaug_block2_random201369 = {
    "annotation_path": "/tos-bjml-video/linhan/Data_Sel_Aug_Net/random_selection/result/sel_groupaug_block2_random201369.json",
    "data_path": "multi"
}
LLAVA_IMAGE_TUNE_sel_aug_block2_random201369 = {
    "annotation_path": "/tos-bjml-video/linhan/Data_Sel_Aug_Net/random_selection/result/sel_aug_block2_random201369.json",
    "data_path": "multi"
}

LLAVA_IMAGE_TUNE_clip006 = {
    "annotation_path": "/tos-bjml-video/linhan/Data_Sel_Aug_Net/indicator_selection/clip_score/ft_json/llava_image_clip_score_0.94_1.0.json",
    "data_path": "multi"
}

LLAVA_IMAGE_TUNE_coincide006={
    "annotation_path": "/tos-bjml-video/linhan/Data_Sel_Aug_Net/COINCIDE_code/scores_multi/llava_image_coincide0.06.json",
    "data_path": "multi"
}

LLAVA_IMAGE_TUNE_lessmm006={
    "annotation_path": "/tos-bjml-video/linhan/Data_Sel_Aug_Net/LESS/selected_data_single_multimodal_tar_validation/gqa_sqa_textvqa/top_p0.06_multi.json",
    "data_path": "multi"
}
LLAVA_IMAGE_TUNE_CHERRYI006={
    "annotation_path": "/tos-bjml-video/linhan/Data_Sel_Aug_Net/cherry_selection/ft_json/llava_image_cherry_I_0.06.json",
    "data_path": "multi"
}
NLP_TUNE = {
    "annotation_path": "/fs-computility/video/shared/linhan/code/video_llava/Video-LLaVA/data_ft/ft_json/nlp_tune.json",
    "data_path": ""
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "llava_image_tune": LLAVA_IMAGE_TUNE,
    "nlp_tune": NLP_TUNE,
    "videochatgpt_random6": VIDEOCHATGPT_random6,
    "llava_image_tune_random6": LLAVA_IMAGE_TUNE_random6,  
    "llava_image_tune_sel_groupaug_block2_random201369": LLAVA_IMAGE_TUNE_sel_groupaug_block2_random201369,
    "llava_image_tune_sel_aug_block2_random201369": LLAVA_IMAGE_TUNE_sel_aug_block2_random201369,
    "llava_image_tune_clip006": LLAVA_IMAGE_TUNE_clip006,
    "llava_image_tune_coincide006": LLAVA_IMAGE_TUNE_coincide006,
    "llava_image_tune_lessmm006": LLAVA_IMAGE_TUNE_lessmm006,
    "llava_image_tune_cherryi006": LLAVA_IMAGE_TUNE_CHERRYI006,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        # sampling_rate = parse_sampling_rate(dataset_name)
        sampling_rate = 1
        # dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        print(f"ds_name:{dataset_name}")
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
