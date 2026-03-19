TASKS = {
    "task1": ["nodule", "cyst", "mass", "fluid collection", "normal appearance"],
    "task2": ["Abdomen and retroperitoneum","Urinary Tract and male reproductive system","Gynaecology","Head and Neck",
              "Breast and Axilla","Musculoskeletal Joints and Tendons","Thorax","Pediatrics","Peripheral vessels"],
    "task3": ["Liver","Gallbladder and bile ducts","Pancreas","Spleen","Appendix","Gastrointestinal tract",
              "Peritoneum mesentery and omentum","Retroperitoneum and great vessels","Adrenal glands","Abdominal wall",
              "Kidney and ureter","Bladder","Scrotum","Penis and perineum","Uterus","Adnexa","Vagina",
              "Thyroid gland","Parathyroid glands","Salivary glands","Lymph nodes","Ocular","Ear","Larynx",
              "Breast","Axilla","Shoulder","Elbow","Wrist and carpus","Fingers","Hip groin and buttock","Knee","Ankle","Foot",
              "Peripheral nerves","Soft tissues","Skull","Pulmonary","Pleural space","Heart and mediastinum","Thoracic wall",
              "Pediatric abdomen and retroperitoneum","Pediatric urinary tract","Pediatric scrotum",
              "Pediatric gynaecological pathology and infant breast","Pediatric head and neck","Neonatal brain and spine",
              "Infant hip and knee","Pediatric thorax","Peripheral arteries","Peripheral veins","Dialysis fistula"],
    "task4": ["round","oval","lobulated","tubular/linear","nodular","flattened","irregular"],
    "task5": ["well-defined","ill-defined/indistinct"],
    "task6": ["anechoic","hypoechoic","isoechoic","hyperechoic","mixed echogenicity"],
    "task7": ["cystic components","calcifications","septations","solid components","mixed cystic and solid mass","complex cystic solid"],
    "task8": ["enhancement","shadowing"],
    "task9": ['reduced/diminished vascularity','normal/regular vascularity','no vascularity','increased vascularity','indeterminate/inhomogeneous vascularity'],
}

def build_vocabs():
    diagnosis_vocab = {}
    descriptor_vocab = {}
    for idx, tag in enumerate(TASKS["task1"]):
        diagnosis_vocab[tag] = idx
    offset = 0
    for tk in ["task2","task3","task4","task5","task6","task7","task8","task9"]:
        for tag in TASKS[tk]:
            key = f"{tk}::{tag}"
            descriptor_vocab[key] = offset
            offset += 1
    return diagnosis_vocab, descriptor_vocab

DIAGNOSIS_VOCAB, DESCRIPTOR_VOCAB = build_vocabs()
UNK_DIAG_ID = len(DIAGNOSIS_VOCAB)
UNK_DESC_ID = len(DESCRIPTOR_VOCAB)