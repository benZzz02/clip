# zero_shot_prompts.py
# -*- coding: utf-8 -*-

"""
Zero-shot prompt templates collected from the SurgLaVi / SurgCLIP paper appendix
(Fig. 13-19 as currently extracted).

Included datasets:
- autolaparo_phase
- grasp_phase
- grasp_step
- grasp_tool
- cholec80_tool
- bern_bypass70_phase
- stras_bypass70_phase
- sar_rarp50_phase

Utility functions:
- list_datasets()
- get_label_names(dataset_name)
- get_prompts(dataset_name)
- get_prompt_dict(dataset_name)
- build_text_inputs(dataset_name, tokenizer)

Notes:
1. These prompts are intended for CLIP-style zero-shot evaluation.
2. Make sure your dataset label order matches the order returned by get_label_names().
3. Fig. 12 prompts are not included here yet.
"""


DATASET_PROMPTS = {
    "autolaparo_phase": {
        "Preparation": (
            "The surgical team introduces the laparoscope and trocars into the patient's abdominal cavity. The image shows an insufflated abdomen, with the camera passing through a trocar port. You may see the abdominal wall, port sites, and internal organs being inspected for initial positioning."
        ),
        "Dividing Ligament and Peritoneum": (
            "The surgeon uses a laparoscopic dissector or hook to carefully separate the round ligament and incise the peritoneum. The instruments interact with thin,translucent membranes and connective tissue, exposing anatomical landmarks like the uterus and fallopian tubes"
        ),
        "Dividing Uterine Vessels and Ligament": (
            "The image displays the surgeon securing and cutting the uterine arteries and ligaments. Vascular clips or bipolar cautery tools are often used, and the view may show controlled bleeding or coagulation near the uterus and cervix."
        ),
        "Transecting the Vagina": (
            "The surgeon uses an energy device or scalpel to cut around the vaginal cuff, detaching the uterus from the vaginal canal. The view is focused at the lower end of the uterus, often showing circular incisions and separation of deep tissue layers"
        ),
        "Specimen Removal": (
            "The uterus, now fully detached, is placed in a specimen retrieval bag and removed through a trocar or vaginal route. The image may show the bag being opened, tissue being pulled, or the uterus partially exiting through a port or canal."
        ),
        "Suturing": (
            "The surgeon inserts a needle and uses a laparoscopic needle driver to suture the vaginal cuff. The view includes precise needle movement through soft tissue, thread pulling, and tying knots within the pelvic cavity, often under close-up magnification."
        ),
        "Washing": (
            "The surgical field is flushed with saline using a laparoscopic irrigation-suction device. You see fluid dispersion, clearing of blood or debris, and suctioning of pooled liquid to improve visibility and cleanliness before closure."
        ),
    },

    "grasp_phase": {
    "Idle": "The surgical instruments are static or resting outside the patient, with no active dissection or movement occurring in the field of view.",
    "Left pelvic isolated lymphadenectomy": "Surgical instruments are dissecting and removing lymphatic tissue from the left pelvic side, focusing around the iliac vessels and obturator nerve",
    "Right pelvic isolated lymphadenectomy": "Dissection is occurring on the right pelvic region with tools navigating around the right iliac artery and vein, isolating lymph nodes near the obturator fossa",
    "Developing the Space of Retzius": "Surgeons are entering the space between the pubic bone and the bladder, pushing down the peritoneum and exposing the anterior pelvic structures.",
    "Ligation of the deep dorsal venous complex": "A vessel-sealing or suturing device is being used to control the deep dorsal venous complex, located at the apex of the prostate.",
    "Bladder neck identification and transection": "The bladder neck is being visualized and transected, separating it from the prostate, with dissection along the midline plane.",
    "Seminal vesicle dissection": "Surgeons expose and dissect the seminal vesicles from surrounding connective tissue, typically located posterior to the bladder.",
    "Development of the plane between the prostate and rectum": "A sharp or blunt instrument is developing a surgical plane between the prostate and the rectum, involving the Denonvilliers fascia.",
    "Prostatic pedicle control": "Surgical clips or energy devices are applied to the lateral pedicles of the prostate to control bleeding and facilitate mobilization.",
    "Severing of the prostate from the urethra": "The urethra is being incised at the prostate apex, with instruments clearly cutting or separating the urethral tissue from the prostate.",
    "Bladder neck reconstruction": "Sutures are being passed between the urethra and bladder neck, visualizing reconstruction with knot tying and tissue approximation.",
},

    "grasp_step": {
    "Idle": "No active motion; instruments may be parked or camera is observing without manipulation.",
    "Identification and dissection of the Iliac vein and artery": "The iliac artery and vein are being visually identified and carefully dissected to expose their full course.",
    "Cutting and dissection of the external iliac vein's lymph node": "Lymphatic tissue surrounding the external iliac vessels is being dissected, often with monopolar or bipolar energy.",
    "Obturator nerve and vessel path identification, dissection, and cutting of the obturator lymph nodes": "Surgeons expose the obturator nerve and remove surrounding lymph nodes using precise dissection near critical vessels.",
    "Insert the lymph nodes in retrieval bags": "Collected lymphatic tissue is being placed into retrieval bags using graspers or suction.",
    "Prevessical dissection": "Tissue anterior to the bladder is dissected to expose the Retzius space, moving fat and fascia.",
    "Ligation of the dorsal venous complex": "A suture or stapling device is controlling the dorsal venous complex at the prostate apex.",
    "Prostate dissection until the levator ani": "Instruments are dissecting around the prostate from the pelvic floor muscles, progressing toward levator ani.",
    "Seminal vesicle dissection": "The surgeon is isolating and freeing the seminal vesicles from adjacent tissues, often with delicate movements.",
    "Dissection of Denonviliers’ fascia": "Sharp or blunt dissection of the fascia between the prostate and rectum to define the posterior surgical plane.",
    "Cut the tissue between the prostate and the urethra": "The surgeon cuts the tissue connection between the prostate and urethra, typically at the apex.",
    "Hold prostate": "A grasper or forceps is holding the prostate to maintain tension or facilitate dissection.",
    "Insert prostate in retrieval bag": "The prostate specimen is being inserted into a retrieval bag for extraction.",
    "Pass suture to the urethra": "A needle is passed through the urethral stump to initiate urethrovesical anastomosis.",
    "Pass suture to the bladder neck": "A needle driver passes a suture through the bladder neck for reconstruction.",
    "Pull suture": "The surgeon pulls a suture thread to tighten or align tissues for anastomosis.",
    "Tie suture": "A knot is being tied with instruments to secure the suture, either manually or using a knot pusher.",
    "Suction": "A suction instrument is removing blood or fluid from the surgical field.",
    "Cut suture or tissue": "Scissors or energy device is actively cutting tissue or suture material.",
    "Cut between the prostate and bladder neck": "A clear incision is made between the bladder neck and prostate using energy or scissors.",
    "Vascular pedicle control": "Metal clips or vessel sealers are applied to vascular pedicles adjacent to the prostate.",
},

    "grasp_instrument": {
    "Bipolar Forceps": "Robotic surgical instrument bipolar forceps is visible in the surgical field as thin, metallic tweezers with dual parallel tips.",
    "Prograsp Forceps": "Robotic surgical instrument prograsp forceps are present when serrated or toothed metallic jaws are visible within the surgical field. The instrument appears as a laparoscopic tool with distinctive interlocking ridged surfaces.",
    "Large Needle Driver": "Robotic surgical instrument A large needle driver is visible when robust metallic jaws with cross-hatched or diamond-pattern surfaces are seen.",
    "Monopolar Curved Scissors": "Robotic surgical instrument Monopolar curved scissors are present when curved metallic blades are visible actively cutting through tissue or positioned for dissection. The instrument appears with distinctive curved cutting edges.",
    "Suction Instrument": "A suction instrument is visible when a hollow, tube-like device with a tapered tip is actively removing fluid or blood from the surgical field. The instrument appears as a metallic wand.",
    "Clip Applier": "A clip applier is present when a specialized instrument with a cartridge-loading mechanism is visible positioning or deploying small metallic clips onto blood vessels.",
    "Laparoscopic Grasper": "A laparoscopic grasper is visible when a long, slender instrument with articulating jaws is seen.",
},

    "cholec80_tool": {
        "Grasper": "Use grasper or cautery forcep to grasp it.",
        "Bipolar": "Use bipolar instrument to coagulate and clean the bleeding.",
        "Hook": "Use hook to dissect it.",
        "Scissors": "Use scissor to cut.",
        "Clipper": "Use clipper to clip it.",
        "Irrigator": "Use irrigator to suck it.",
        "Specimen Bag": "Use the specimen bag to wrap it.",
    },

    "bern_bypass70_phase": {
    "Preparation": "In preparation phase I insert trocars to the abdominal cavity and expose of the operating field.",
    "Gastric Pouch Creation": "I cut the fat tissue and open retrogastric window at stomach.",
    "Omentum Division": "I grasp and lift the omentum and divide it.",
    "Gastrojejunal Anastomosis": "I see the proximal jejunum and determine the length of the biliary limb. I open the distal jejunum and create the gastrojejunostomy using a stapler. I reinforcement of the gastrojejunostomy with an additional suture.",
    "Anastomosis Test": "I place the retractor and move the gastric tube and detect any leakage of the gastrojejunostomy.",
    "Jejunal Separation": "I open the mesentery to facilitate the introduction of the stapler and transect the jejunum proximal.",
    "Petersen Space Closure": "I expose between the alimentary limb and the transverse colon and close it with sutures.",
    "Jejunojejunal Anastomosis": "I bring together the proximal and distal segments of the jejunum, create an opening in each segment, and perform a side-to-side jejunojejunostomy using a stapler or sutures.",
    "Mesenteric Defect Closure": "I expose the mesenteric defect and then close it by stitches.",
    "Cleaning Coagulation": "In clean and coagulation phase I use suction and irrigation to clear the surgical field and coagulate bleeding vessels.",
    "Disassembling": "I remove the instruments, retractor, ports, and camera.",
},

"stras_bypass70_phase": {
    "Preparation": "In preparation phase I insert trocars to the abdominal cavity and expose of the operating field.",
    "Gastric Pouch Creation": "I cut the fat tissue and open retrogastric window at stomach.",
    "Omentum Division": "I grasp and lift the omentum and divide it.",
    "Gastrojejunal Anastomosis": "I see the proximal jejunum and determine the length of the biliary limb. I open the distal jejunum and create the gastrojejunostomy using a stapler. I reinforcement of the gastrojejunostomy with an additional suture.",
    "Anastomosis Test": "I place the retractor and move the gastric tube and detect any leakage of the gastrojejunostomy.",
    "Jejunal Separation": "I open the mesentery to facilitate the introduction of the stapler and transect the jejunum proximal.",
    "Petersen Space Closure": "I expose between the alimentary limb and the transverse colon and close it with sutures.",
    "Jejunojejunal Anastomosis": "I bring together the proximal and distal segments of the jejunum, create an opening in each segment, and perform a side-to-side jejunojejunostomy using a stapler or sutures.",
    "Mesenteric Defect Closure": "I expose the mesenteric defect and then close it by stitches.",
    "Cleaning Coagulation": "In clean and coagulation phase I use suction and irrigation to clear the surgical field and coagulate bleeding vessels.",
    "Disassembling": "I remove the instruments, retractor, ports, and camera.",
},

    "sar_rarp50_phase": {
        "Other": (
            "A scene with activities unrelated to the handling or manipulation of a needle or suture."
        ),
        "Picking up the needle": (
            "Picking up the needle: An instrument approaching and grasping a needle."
        ),
        "Positioning the needle tip": (
            "Positioning the needle tip: A needle being carefully aligned or adjusted, "
            "with the tip directed towards a target area."
        ),
        "Pushing the needle through the tissue": (
            "Pushing the needle through the tissue: A needle being driven through a piece of tissue."
        ),
        "Pulling the needle out of the tissue": (
            "Pulling the needle out of the tissue: Pulling a needle backwards, retiring it from a piece of tissue."
        ),
        "Tying a knot": (
            "Tying a knot: Instrument manipulating suture material to form a loop and tighten it into a secure knot."
        ),
        "Cutting the suture": (
            "Cutting the suture: Scissors tool being used to cut the suture material close to the knot."
        ),
        "Returning/dropping the needle": (
            "Returning/dropping the needle action: A needle being placed back onto a holder or released to fall."
        ),
        },
    "cholec80_phase": {
    "Preparation": "Prepares for surgery by inserting trocars into the patient's abdominal cavity",
    "Calot Triangle Dissection": "Employs grasper and hook during calot triangle dissection, manipulating gallbladder to reveal hepatic triangle, cystic duct and cystic artery",
    "Clipping Cutting": "Utilizes clipper to secure cystic duct and artery, followed by precise dissection using scissors",
    "Gallbladder Dissection": "Utilizes a hook to dissect the connective tissue during the dissection phase, separating gallbladder from the liver",
    "Gallbladder Retraction": "Secures the removed gallbladder in the specimen bag during the packaging phase of the procedure",
    "Cleaning Coagulation": "Employs suction and irrigation techniques to maintain a clear surgical field during the clean and coagulation phase, simultaneously coagulating bleeding vessels",
    "Gallbladder Packaging": "Handles the specimen bag during the retraction phases, carefully extracting it from the trocar",

        
    },
}


def list_datasets():
    """Return all supported dataset names."""
    return list(DATASET_PROMPTS.keys())


def get_label_names(dataset_name: str):
    """Return class names in the defined label order."""
    if dataset_name not in DATASET_PROMPTS:
        raise KeyError(f"Unknown dataset: {dataset_name}")
    return list(DATASET_PROMPTS[dataset_name].keys())


def get_prompts(dataset_name: str):
    """Return prompt strings in the defined label order."""
    if dataset_name not in DATASET_PROMPTS:
        raise KeyError(f"Unknown dataset: {dataset_name}")
    return list(DATASET_PROMPTS[dataset_name].values())


def get_prompt_dict(dataset_name: str):
    """Return the {label_name: prompt} dictionary."""
    if dataset_name not in DATASET_PROMPTS:
        raise KeyError(f"Unknown dataset: {dataset_name}")
    return DATASET_PROMPTS[dataset_name]


def build_text_inputs(dataset_name: str, tokenizer):
    """
    Tokenize prompts for CLIP-style zero-shot evaluation.

    Example:
        from zero_shot_prompts import build_text_inputs
        text_tokens = build_text_inputs("grasp_phase", tokenizer)
    """
    prompts = get_prompts(dataset_name)
    return tokenizer(prompts)


def build_text_features(dataset_name: str, tokenizer, text_encoder, device="cuda", normalize=True):
    """
    Build text embeddings from prompts.

    This helper tries to work with common CLIP-style tokenizers and text encoders.

    Example:
        text_features = build_text_features(
            dataset_name="grasp_phase",
            tokenizer=tokenizer,
            text_encoder=model.encode_text,
            device="cuda"
        )

    Notes:
    - If tokenizer returns a dict (e.g. HuggingFace tokenizer), this function handles it.
    - text_encoder can be:
        1) a full text model forward
        2) a CLIP-style encode_text function
    """
    prompts = get_prompts(dataset_name)
    tokens = tokenizer(prompts)

    if isinstance(tokens, dict):
        tokens = {k: v.to(device) for k, v in tokens.items()}
        text_features = text_encoder(**tokens)
    else:
        tokens = tokens.to(device)
        text_features = text_encoder(tokens)

    if normalize:
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features


if __name__ == "__main__":
    print("Supported datasets:")
    for name in list_datasets():
        print(f"- {name}: {len(get_label_names(name))} prompts")

    # Example preview
    dataset_name = "autolaparo_phase"
    print(f"\nExample dataset: {dataset_name}")
    for label, prompt in get_prompt_dict(dataset_name).items():
        print(f"[{label}] {prompt}")