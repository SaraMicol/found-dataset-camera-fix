import os
from os.path import join as p_join


scenes = ["00824_dynamic"]

primary_device="cuda:0"
seed = 0
scene_name = scenes[0]

map_every = 1
keyframe_every = 5
mapping_window_size = 24
tracking_iters = 40
mapping_iters = 60

group_name = "FOUND"
run_name = f"{scene_name}_{seed}"

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every, # Mapping every nth frame
    keyframe_every=keyframe_every, # Keyframe every nth frame
    mapping_window_size=mapping_window_size, # Mapping window size
    report_global_progress_every=500, # Report Global Progress every nth frame
    eval_every=100, # Evaluate every nth frame (at end of SLAM)
    scene_radius_depth_ratio=3, # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    gaussian_distribution="isotropic", # ["isotropic", "anisotropic"] (Isotropic -> Spherical Covariance, Anisotropic -> Ellipsoidal Covariance)
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False, # Save Checkpoints
    checkpoint_interval=100, # Checkpoint Interval
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
        basedir="./data/FOUND",
        gradslam_data_cfg="./configs/data/found.yaml",
        sequence=scene_name,
        # Dimezzata da 480x640 (default): due run indipendenti su 1453
        # frame sono morte di CUDA OOM sempre intorno a ~1.95-1.96M
        # gaussiane (frame 207 e 216), con la GPU tutta per la pipeline,
        # nessuna concorrenza -- non e' stato un incidente, e' il tasso di
        # crescita per-frame che sulla lunghezza intera della sequenza
        # esaurisce i 16GB prima della fine. Il pruning
        # (mapping.pruning_dict) e' pensato per sequenze corte (i config
        # noti usano num_frames=40) e non tiene il passo su 1453 frame.
        # Toccare quella logica stanotte senza poterla validare e' piu'
        # rischioso che ridurre il numero di pixel: meta' risoluzione ->
        # un quarto dei pixel proiettati per frame -> gaussiane nuove
        # aggiunte per frame proporzionalmente ridotte, a parita' di
        # logica di ottimizzazione/pruning. ReplicaDataset (dataset_name
        # 'replica' in found.yaml) fa resize reale qui (vedi
        # basedataset.py::_preprocess_color/_preprocess_depth), quindi il
        # taglio e' effettivo, non solo dichiarato.
        desired_image_height=240,
        desired_image_width=320,
        start=0,
        end=-1,
        # stride=2 (un frame su due) e' la seconda leva contro l'OOM, dopo
        # removal_opacity_threshold=0.3. Misurato: la soglia da sola porta
        # il tasso da ~3945 a ~2371 gaussiane/frame (-40%), ma proiettato
        # su 1453 frame fa ancora ~3.44M, sopra il tetto osservato di
        # ~1.95M. Con stride=2 i frame processati diventano 726 e la
        # proiezione scende a ~1.72M: sotto il tetto, con margine.
        #
        # Perche' stride e NON un taglio di num_frames: basedataset.py:182-188
        # applica lo stride a color/depth/poses insieme, quindi la sequenza
        # resta COMPLETA dall'inizio alla fine, solo campionata a meta'
        # cadenza. La fase dinamica resta esattamente nella stessa
        # posizione relativa (36% della sequenza, frame 258 di 726 invece
        # di 517 di 1453): tutti gli spawn/move/remove restano coperti.
        # Tagliare num_frames a ~290 (l'altra opzione per stare sotto il
        # tetto) avrebbe invece fermato la run PRIMA di frame_begin_update,
        # senza osservare nemmeno un cambiamento.
        stride=2,
        # ATTENZIONE: questi due valori sono espressi in INDICI DEL DATASET
        # GIA' CAMPIONATO (post-stride), non in frame grezzi del disco.
        # Il dataset registrato ha 1453 frame e frame_begin_update=517
        # (dynamic_meta.json), ma con stride=2 basedataset.py tiene un
        # frame su due -> restano 727 item, e l'evento dinamico cade
        # all'indice 258.
        #
        # Perche' conta: il loop di dgsg() itera su range(num_frames) e la
        # riga 969 confronta quel contatore con frame_begin_update. Sono
        # entrambi indici del dataset campionato. Lasciando i valori grezzi:
        #   - num_frames=1453 > 727 item disponibili -> IndexError a fine run;
        #   - frame_begin_update=517 farebbe partire il controllo dinamico
        #     al 71% della sequenza invece che al 35%, mancando gran parte
        #     degli spawn/move/remove da misurare.
        num_frames=727,
        frame_begin_update=258,
        ignore_bad = False,
        use_train_split = True,
    ),
    tracking=dict(
        modify_real_gt_poses=True, # Modify Real GT Poses for Tracking
        use_gt_poses=False, # Use GT Poses for Tracking
        forward_prop=True, # Forward Propagate Poses
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
        sil_thres=0.98, # For Addition of new Gaussians
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
        prune_gaussians=True, # Prune Gaussians during Mapping
        # Reso piu' aggressivo dopo tre OOM su sequenze da 1453 frame (vedi
        # nota su desired_image_height/width sopra). Due tentativi
        # progressivi, entrambi misurati su dati reali della stessa scena:
        #   1) solo dimezzare la risoluzione -> 6242 gaussiane/frame nette,
        #      OOM proiettato al frame ~365.
        #   2) + aumentare la frequenza di pruning (stop_after 20->60,
        #      prune_every 20->10, 7 potature/frame invece di 1) -> NESSUN
        #      cambiamento misurabile (frame 148: 583908 vs 597160
        #      gaussiane della prova precedente). La frequenza non era il
        #      collo di bottiglia: con removal_opacity_threshold=0.005
        #      (sigmoid dell'opacita' < 0.005, quasi trasparente) quasi
        #      nessuna gaussiana la attraversa mai, quindi ripetere il
        #      controllo piu' spesso non cambia chi viene rimosso.
        # Qui si alza la soglia stessa di due ordini di grandezza (0.3):
        # rimuove le gaussiane debolmente opache, non solo quelle
        # praticamente invisibili. Rischio esplicito e non nascosto: una
        # soglia cosi' alta puo' rimuovere anche gaussiane valide non
        # ancora convergenti, con un impatto sulla qualita' della
        # ricostruzione che qui non e' stato validato visivamente --
        # scelta fatta perche' l'alternativa (limitare num_frames a un
        # valore sicuro, ~290 stimati) taglierebbe la sequenza PRIMA di
        # frame_begin_update=517, cioe' prima che inizi la parte dinamica
        # da misurare: uno zero garantito e' peggio di una ricostruzione
        # via via piu' scarna ma completa fino in fondo.
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=0,
            remove_big_after=0,
            stop_after=60,
            prune_every=10,
            removal_opacity_threshold=0.3,
            final_removal_opacity_threshold=0.3,
            reset_opacities=False,
            reset_opacities_every=500, # Doesn't consider iter 0
        ),
        use_gaussian_splatting_densification=False, # Use Gaussian Splatting-based Densification during Mapping
        densify_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=500,
            remove_big_after=3000,
            stop_after=5000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000, # Doesn't consider iter 0
        ),
    ),
    lang=dict(
        use_lang=True,
        detection_model="groundingdino", # ["groundingdino", "yolo"]
        color_book_path="./configs/scannet200.txt",
        yolo_model_path="./models/yolov8l-world.pt",
        grounding_dino_config_path="./submodules/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint_path="./models/groundingdino_swint_ogc.pth",
        ram_model_path="./models/ram_plus_swin_large_14m.pth",
        sam_model_path='./models/sam_l.pt',
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="ollama",
        llm_model="gemma3:4b",
        # DAM-3B disattivato: veniva caricato su cuda:0 da save_objects()
        # mentre GroundingDINO, SAM, RAM, CLIP e le gaussiane erano gia'
        # residenti, ed e' la causa reale di TUTTI gli OOM di questa notte
        # (traceback: dynamic_gsg_real_ssim.py:1064 -> save_objects ->
        # vision_tower.to(cuda) -> 12.4 GiB su 15.47 disponibili). Il crash
        # cadeva sempre a frame_begin_update perche' e' li' che il loop
        # chiama save_objects la prima volta, non per le gaussiane, che nei
        # test erano stabili a ~150k e in calo.
        # Serve solo a generare obj['description'] e a riscrivere
        # obj['category'] via LLM: nessuno dei due entra nelle metriche
        # spawn/move/remove, che leggono idx/category/centroid/clip_ft da
        # graph_stream.jsonl (scritto da log_graph_state, indipendente).
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
        update_gs_ratio_threshold = 0.9,
    ),
    viz=dict(
        render_mode='color', # ['color', 'depth' or 'centers']
        follow_walk=True, # Segue percorso smussato a altezza occhi invece della GT raw stop-and-go
        walk_smooth_window=15, # Finestra media mobile sulle posizioni (frame)
        walk_eye_height=None, # Quota occhi nel world (None = mediana traiettoria); es. 1.6
        walk_level_camera=True, # Azzera pitch/roll, tiene solo yaw: sguardo orizzontale da persona in piedi
        offset_first_viz_cam=True, # Offsets the view camera back by 0.5 units along the view direction (For Final Recon Viz)
        show_sil=False, # Show Silhouette instead of RGB
        show_bg=True, # Show Background
        visualize_cams=False, # Visualize Camera Frustums and Trajectory
        viz_w=640, viz_h=480,
        viz_near=0.01, viz_far=100.0,
        view_scale=1,
        viz_fps=5, # FPS for Online Recon Viz
        enter_interactive_post_online=True, # Enter Interactive Mode after Online Recon Viz
        no_clip=False, # If set, the CLIP model will not init for fast debugging.
        clip_model_path='./models/open_clip_pytorch_model.bin',
        variables_path=f"./experiments/FOUND/{run_name}/variables.npz",
        keyframe_list_path=f"./experiments/FOUND/{run_name}/keyframelist.pkl.gz",
    ),
)
