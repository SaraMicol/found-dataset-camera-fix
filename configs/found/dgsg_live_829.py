# Config LIVE per la scena 829 -- gemello di dgsg_live.py (scena 824).
# Generato da quel file cambiando SOLO scene_name, cosi' i due esperimenti
# restano confrontabili: stesse soglie, stesse iterazioni, stessi fix
# (map_every=2, tracking/mapping 20/30, use_dam=False).
# I commenti sotto citano misure fatte sulla 824: valgono come motivazione
# della scelta, non come dato della 829.
import os
from os.path import join as p_join


# Non e' un nome di cartella da risolvere (RosLiveDataset non legge file):
# resta solo per costruire run_name/output_dir in modo leggibile, coerente
# con lo schema degli altri config found/*.py.
scene_name = "00829_live"

primary_device="cuda:0"
seed = 0

# map_every alzato da 1 a 2 dopo l'OOM del 2026-09-12 17:14 (CUDA
# out-of-memory nell'encoder SAM al frame 158, run mai completata: vedi
# logs/pipeline824live_20260912_171457.log:7579). Con map_every=1 ogni
# frame passa per GroundingDINO+SAM+CLIP oltre al training gaussiano sulla
# stessa GPU da 15.47 GiB; a 2 la pressione di picco si dimezza. Lo stride
# del dataset non si applica qui (bridge live, non file offline): questa
# e' l'unica leva di campionamento disponibile insieme a use_dam=False.
map_every = 2
keyframe_every = 5
mapping_window_size = 24
# Dimezzati (40/60 -> 20/30) il 2026-09-12 per la run live: misurato
# 3.26 s/frame, di cui la parte dominante era proprio l'ottimizzazione
# (tracking 40 iter ~0.5s + mapping 60 iter ~1.1s, contro 0.4s per il
# render di 115 oggetti). Con 1453 frame significava ~68 minuti.
# Le metriche spawn/move/remove dipendono da detection, category e
# centroidi degli oggetti -- non dalla finezza del rendering gaussiano --
# quindi il dimezzamento accelera senza toccare cio' che si sta validando.
# Per una ricostruzione di qualita' (rendering, PSNR) tornare a 40/60.
tracking_iters = 20
mapping_iters = 30

group_name = "FOUND"
run_name = f"{scene_name}_{seed}"

config = dict(
    # Disattivato per la run headless: senza finestra OpenCV la pipeline non
    # deve aprire un display (utile su questa macchina/sessione, e riduce il
    # lavoro per frame). Il confronto con la GT continua a scrivere
    # 00829_live.json via watch_live_progress.py, indipendente dal viewer.
    live_viewer=False,
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every,
    keyframe_every=keyframe_every,
    mapping_window_size=mapping_window_size,
    report_global_progress_every=500,
    eval_every=100,
    scene_radius_depth_ratio=3,
    mean_sq_dist_method="projective",
    gaussian_distribution="isotropic",
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False,
    checkpoint_interval=100,
    use_wandb=False,
    whether_to_update = True,
    wandb=dict(
        entity="ICR-Lab",
        project="Dynamic-GSG",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        # basedir/sequence non risolvono nessun file per RosLiveDataset
        # (bypassato -- vedi datasets/gradslam_datasets/ros_live.py), ma
        # scripts/dynamic_gsg_real_ssim.py li legge comunque come stringhe
        # prima di chiamare get_dataset(): devono esistere nel config.
        basedir="./data/FOUND",
        sequence=scene_name,
        gradslam_data_cfg="./configs/data/ros_live.yaml",
        desired_image_height=480,
        desired_image_width=640,
        start=0,
        end=-1,
        stride=1,
        # DEVE combaciare con configs/data/ros_live.yaml:num_frames e con
        # quanti frame lo script di cammino live pubblicherà davvero (vedi
        # commenti in entrambi i file). frame_begin_update qui è quello
        # atteso per lo script FOUND completo (household_experiments_scene_824.json,
        # 1453 frame totali, cambiamento dinamico dal frame 517 misurato in
        # un run già registrato su file) -- da aggiornare se si usa
        # --max-steps per un test più corto.
        num_frames=1453,
        frame_begin_update=517,
        ignore_bad = False,
        use_train_split = True,
    ),
    tracking=dict(
        modify_real_gt_poses=True,
        use_gt_poses=False,
        forward_prop=True,
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.5,
        use_l1=True,
        ignore_outlier_depth_loss=True,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
        ),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            features=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.0004,
            cam_trans=0.002,
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.98,
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
            features=0.5,
        ),
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.0025,
            features=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
        ),
        prune_gaussians=True,
        # Soglia alzata dopo gli OOM della notte del 2026-09-11/12 su
        # dgsg_dynamic.py (stessa pipeline, dataset offline): a 0.005 quasi
        # nessuna gaussiana viene mai rimossa. Vedi configs/found/dgsg_dynamic.py
        # per la misura completa (0.005->0.3 taglia il tasso di crescita di
        # circa il 40%). Qui lo stride non si applica (il bridge riceve un
        # frame alla volta in tempo reale, non un elenco di file da
        # campionare), quindi questa e' l'unica leva insieme a use_dam=False
        # sotto.
        pruning_dict=dict(
            start_after=0,
            remove_big_after=0,
            stop_after=60,
            prune_every=10,
            removal_opacity_threshold=0.3,
            final_removal_opacity_threshold=0.3,
            reset_opacities=False,
            reset_opacities_every=500,
        ),
        use_gaussian_splatting_densification=False,
        densify_dict=dict(
            start_after=500,
            remove_big_after=3000,
            stop_after=5000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000,
        ),
    ),
    lang=dict(
        use_lang=True,
        detection_model="groundingdino",
        color_book_path="./configs/scannet200.txt",
        yolo_model_path="./models/yolov8l-world.pt",
        grounding_dino_config_path="./submodules/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint_path="./models/groundingdino_swint_ogc.pth",
        ram_model_path="./models/ram_plus_swin_large_14m.pth",
        sam_model_path='./models/sam_l.pt',
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="ollama",
        llm_model="gemma3:4b",
        # DAM-3B disattivato: e' la causa reale di TUTTI gli OOM osservati
        # sulla stessa pipeline (dataset offline, notte del 2026-09-11/12).
        # save_objects() lo caricava su cuda:0 sopra GroundingDINO+SAM+RAM+
        # CLIP gia' residenti (12.4 GiB su 15.47 disponibili). Serve solo a
        # generare obj['description'] e a riscrivere obj['category'] via
        # LLM: nessuno dei due entra nelle metriche spawn/move/remove, che
        # leggono idx/category/centroid/clip_ft da graph_stream.jsonl
        # (scritto da log_graph_state, indipendente da DAM).
        use_dam=False,
        dam_model_path='nvidia/DAM-3B',
        dam_conv_mode="v1",
        dam_prompt_mode="focal_prompt",
        sys_prompt_file="./configs/prompts/parsing_query.txt",
        obj_prompt_file="./configs/prompts/parsing_objects.txt",
        obj_caption_file="./configs/prompts/parsing_objects_caption.txt",
        clip_model_path='./models/open_clip_pytorch_model.bin',
        classes_file="./configs/scannet200_classes.txt",
        bg_classes=["wall", "floor", "ceiling"],
        skip_bg=True,
        mask_area_threshold=10,
        max_bbox_area_ratio=0.6,
        mask_conf_threshold=0.4,
        merge_overlap_thresh=0.7,
        merge_visual_sim_thresh=0.7,
        similarity_bias = 0.0,
        similarity_threshold = 0.55,
        update_gs_num_threshold = 500,
        # 0.9 -> 0.7 dopo la misura del 2026-09-12 sulla scena 824: un
        # oggetto nuovo riceve gaussiane solo se almeno questa frazione
        # delle gaussiane piu' vicine e' libera o gia' sua (vedi
        # update_curr_objects_gaussians). A 0.9, 15 scarti su 16 avevano
        # ratio 0.71-0.89 -- oggetti veri con migliaia di punti, respinti
        # per pochi centesimi perche' in una scena affollata condividono
        # sempre qualche gaussiana coi vicini. Un oggetto scartato resta
        # senza gaussiane, si rasterizza a 0 pixel, sparisce dal pool di
        # confronto e la detection successiva lo ricrea: e' cosi' che
        # nascevano i 17 "pillow" tutti con detections=1.
        update_gs_ratio_threshold = 0.7,
    ),
    viz=dict(
        render_mode='color',
        follow_walk=True,
        walk_smooth_window=15,
        walk_eye_height=None,
        walk_level_camera=True,
        offset_first_viz_cam=True,
        show_sil=False,
        show_bg=True,
        visualize_cams=False,
        viz_w=640, viz_h=480,
        viz_near=0.01, viz_far=100.0,
        view_scale=1,
        viz_fps=5,
        enter_interactive_post_online=True,
        no_clip=False,
        clip_model_path='./models/open_clip_pytorch_model.bin',
        variables_path=f"./experiments/FOUND/{run_name}/variables.npz",
        keyframe_list_path=f"./experiments/FOUND/{run_name}/keyframelist.pkl.gz",
    ),
)
