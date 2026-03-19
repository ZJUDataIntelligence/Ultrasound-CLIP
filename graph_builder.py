import torch
import dgl
from .tag_vocab import DIAGNOSIS_VOCAB, DESCRIPTOR_VOCAB, UNK_DIAG_ID, UNK_DESC_ID

def build_single_sample_graph(full_data_dict, image_key):
    rec = full_data_dict.get(image_key, {})
    task_keys = [
        "Diagnosis",
        "Body_system_level",
        "Organ_level",
        "Shape",
        "Margins", 
        "Echogenicity",
        "InternalCharacteristics",
        "PosteriorAcoustics",
        "Vascularity",
    ]
    diagnosis = rec.get("Diagnosis", [])
    if not isinstance(diagnosis, list):
        diagnosis = [diagnosis] if diagnosis else []
    descriptors = []
    for tk in task_keys[1:]:
        tags = rec.get(tk, [])
        if not isinstance(tags, list):
            tags = [tags] if tags else []
        descriptors.extend(tags)
    diagnosis_list = sorted(list(set(diagnosis)))
    descriptor_list = sorted(list(set(descriptors)))
    if not diagnosis_list and not descriptor_list:
        g = dgl.heterograph(
            {
                ('diagnosis', 'described_by', 'descriptor'): (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)),
                ('descriptor', 'rev_described_by', 'diagnosis'): (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
            },
            num_nodes_dict={'diagnosis': 1, 'descriptor': 1}
        )
        g.nodes['diagnosis'].data['tid'] = torch.tensor([UNK_DIAG_ID], dtype=torch.long)
        g.nodes['descriptor'].data['tid'] = torch.tensor([UNK_DESC_ID], dtype=torch.long)
        return g
    
    diagnosis2id = {d: i for i, d in enumerate(diagnosis_list)}
    descriptor2id = {d: i for i, d in enumerate(descriptor_list)}
    
    src_d, dst_desc = [], []
    for d in diagnosis:
        for desc in descriptors:
            if d in diagnosis2id and desc in descriptor2id:
                src_d.append(diagnosis2id[d])
                dst_desc.append(descriptor2id[desc])
    
    if len(src_d) == 0:
        g = dgl.heterograph(
            {
                ('diagnosis', 'described_by', 'descriptor'): (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)),
                ('descriptor', 'rev_described_by', 'diagnosis'): (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
            },
            num_nodes_dict={'diagnosis': max(1, len(diagnosis_list)), 'descriptor': max(1, len(descriptor_list))}
        )
        if len(diagnosis_list) > 0:
            diag_tid = [DIAGNOSIS_VOCAB.get(d, UNK_DIAG_ID) for d in diagnosis_list]
        else:
            diag_tid = [UNK_DIAG_ID]
        if len(descriptor_list) > 0:
            
            from .tag_vocab import TASKS
            tag2task = {}
            for tk in ["task2","task3","task4","task5","task6","task7","task8","task9"]:
                for t in TASKS[tk]:
                    if t not in tag2task:
                        tag2task[t] = tk
            desc_tid = [DESCRIPTOR_VOCAB.get(f"{tag2task.get(desc, 'task2')}::{desc}", UNK_DESC_ID) for desc in descriptor_list]
        else:
            desc_tid = [UNK_DESC_ID]
        g.nodes['diagnosis'].data['tid'] = torch.tensor(diag_tid, dtype=torch.long)
        g.nodes['descriptor'].data['tid'] = torch.tensor(desc_tid, dtype=torch.long)
        return g

    data_dict = {
        ('diagnosis', 'described_by', 'descriptor'): (torch.tensor(src_d, dtype=torch.long), torch.tensor(dst_desc, dtype=torch.long)),
        ('descriptor', 'rev_described_by', 'diagnosis'): (torch.tensor(dst_desc, dtype=torch.long), torch.tensor(src_d, dtype=torch.long)),
    }
    
    g = dgl.heterograph(
        data_dict,
        num_nodes_dict={'diagnosis': len(diagnosis_list), 'descriptor': len(descriptor_list)}
    )


    diag_tid = [DIAGNOSIS_VOCAB.get(d, UNK_DIAG_ID) for d in diagnosis_list] if len(diagnosis_list)>0 else [UNK_DIAG_ID]
    from .tag_vocab import TASKS
    tag2task = {}
    for tk in ["task2","task3","task4","task5","task6","task7","task8","task9"]:
        for t in TASKS[tk]:
            if t not in tag2task:
                tag2task[t] = tk
    desc_tid = [DESCRIPTOR_VOCAB.get(f"{tag2task.get(desc, 'task2')}::{desc}", UNK_DESC_ID) for desc in descriptor_list] if len(descriptor_list)>0 else [UNK_DESC_ID]

    g.nodes['diagnosis'].data['tid'] = torch.tensor(diag_tid, dtype=torch.long)
    g.nodes['descriptor'].data['tid'] = torch.tensor(desc_tid, dtype=torch.long)

    return g

def build_hetero_graph_from_data(full_data, image_keys):

    full_data_dict = {rec['media_name']: rec for rec in full_data}
    

    graphs = []
    for image_key in image_keys:
        try:
            graph = build_single_sample_graph(full_data_dict, image_key)
            graphs.append(graph)
        except Exception as e:

            empty_graph = dgl.heterograph(
                {
                    ('diagnosis', 'described_by', 'descriptor'): (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)),
                    ('descriptor', 'rev_described_by', 'diagnosis'): (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
                },
                num_nodes_dict={'diagnosis': 1, 'descriptor': 1}
            )
            graphs.append(empty_graph)

    try:
        batched_graph = dgl.batch(graphs)
        return batched_graph
    except Exception as e:
        return None