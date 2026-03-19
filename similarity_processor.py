import logging
import numpy as np
import torch
import json
import os
from pathlib import Path

def load_matrix_from_upper_triangle(filename):
    data = np.load(filename, allow_pickle=True)
    matrix = data['matrix']
    labels = data['labels'] if 'labels' in data else None
    
    return matrix, labels

class SimilarityMatrixProcessor:
    
    def __init__(self, similarity_matrices_dir, full_data_file=None):

        self.similarity_matrices_dir = Path(similarity_matrices_dir)
        
        assert full_data_file is not None, "must provide full_data_file"
        
        logging.info("load...")
        self.sim_matrices = {}
        self.tag_labels = {}
        
        for task_id in range(1, 10):
            matrix_file = self.similarity_matrices_dir / f"task{task_id}_tag_normalized.npz"
            
            if matrix_file.exists():
                try:
                    
                    matrix, labels = load_matrix_from_upper_triangle(matrix_file)
                    self.sim_matrices[task_id-1] = matrix
                    self.tag_labels[task_id-1] = labels if labels is not None else []
                except Exception as e:
                    
                    raise
            else:
                raise FileNotFoundError(f"similarity matrix file not found: {matrix_file}")

        self.tag_indices = {}
        for task_id in range(9):
            tag_list = list(self.tag_labels[task_id])
            self.tag_indices[task_id] = {tag: idx for idx, tag in enumerate(tag_list)}
        
        with open(full_data_file, "r") as f:
            self.full_data = json.load(f)
   
        self.full_data_dict = {}
        self.full_data_dict_alt = {}
        self.full_data_dict_no_ext = {}
        
        for record in self.full_data:
            media_name = record['media_name']
            self.full_data_dict[media_name] = record
            
            if media_name.endswith('.jpg'):
                self.full_data_dict_no_ext[media_name[:-4]] = record
            else:
                self.full_data_dict_alt[media_name + '.jpg'] = record
        
        self.task_keys = {
            "task1":"Diagnosis",
            "task2":"Body_system_level",
            "task3":"Organ_level",
            "task4":"Shape",
            "task5":"Margins",
            "task6":"Echogenicity",
            "task7":"InternalCharacteristics",
            "task8":"PosteriorAcoustics",
            "task9":"Vascularity"
        }

    
    def get_tags_for_image(self, image_key):
        
        record = (self.full_data_dict.get(image_key) or 
                 self.full_data_dict_alt.get(image_key) or 
                 self.full_data_dict_no_ext.get(image_key))
        
        if record is None:
            
            return {i: [] for i in range(9)}
        
        image_tags = {}
        for i, (task_key, task_name) in enumerate(self.task_keys.items()):
            tags = record.get(task_name, [])

            if not isinstance(tags, list):
                tags = [tags] if tags else []
            image_tags[i] = tags
        
        return image_tags

    def calculate_label_similarity(self, tags1, tags2, task_id):
        
        if not tags1 or not tags2:
            return 0.0
        
        tag_index_map = self.tag_indices[task_id]
        tag_similarity = []
        
        valid_tags1 = [tag for tag in tags1 if tag in tag_index_map]
        valid_tags2 = [tag for tag in tags2 if tag in tag_index_map]
        
        if not valid_tags1 or not valid_tags2:
            return 0.0
        
        for tag in valid_tags1:
            t1_index = tag_index_map[tag]
            for t in valid_tags2:
                t2_index = tag_index_map[t]
                tag_similarity.append(self.sim_matrices[task_id][t1_index, t2_index])
        
        if not tag_similarity:
            return 0.0
        
        return np.mean(tag_similarity)
    
    def calculate_batch_similarity_matrix(self, batch_indices, dataset):
        
        batch_size = len(batch_indices)
        

        image_keys = [dataset.get_image_key(idx) for idx in batch_indices]
        

        batch_tags = []
        for key in image_keys:
            image_tags = self.get_tags_for_image(key)
            batch_tags.append(image_tags)
        

        similarity_matrix = np.zeros((batch_size, batch_size), dtype=np.float32)
        

        for i in range(batch_size):
            similarity_matrix[i, i] = 1.0
            for j in range(i + 1, batch_size):

                task_similarities = []
                for task_id in range(9):
                    tags_i = batch_tags[i][task_id]
                    tags_j = batch_tags[j][task_id]

                    sim = self.calculate_label_similarity(tags_i, tags_j, task_id)
 
                    if not np.isnan(sim) and np.isfinite(sim):
                        task_similarities.append(sim)
                
                if task_similarities:
                    avg_sim = np.mean(task_similarities)
                    similarity_matrix[i, j] = avg_sim
                    similarity_matrix[j, i] = avg_sim
                else:
                    similarity_matrix[i, j] = 0.0
                    similarity_matrix[j, i] = 0.0
        
        return torch.tensor(similarity_matrix, dtype=torch.float32)
    
    def calculate_batch_similarity_matrix_from_paths(self, image_keys, batch_size):
       
        if len(image_keys) != batch_size:
            
            if len(image_keys) > batch_size:
                image_keys = image_keys[:batch_size]
            else:
                last_key = image_keys[-1] if image_keys else "unknown"
                image_keys.extend([last_key] * (batch_size - len(image_keys)))
        
        batch_tags = []
        for key in image_keys:
            image_tags = self.get_tags_for_image(key)
            batch_tags.append(image_tags)
        
        similarity_matrix = np.zeros((batch_size, batch_size), dtype=np.float32)
        
        for i in range(batch_size):
            similarity_matrix[i, i] = 1.0
            for j in range(i + 1, batch_size):
                task_similarities = []
                for task_id in range(9):
                    tags_i = batch_tags[i][task_id]
                    tags_j = batch_tags[j][task_id]
                    
                    sim = self.calculate_label_similarity(tags_i, tags_j, task_id)
                    
                    if not np.isnan(sim) and np.isfinite(sim):
                        task_similarities.append(sim)
                
                if task_similarities:
                    avg_sim = np.mean(task_similarities)
                    similarity_matrix[i, j] = avg_sim
                    similarity_matrix[j, i] = avg_sim
                else:
                    similarity_matrix[i, j] = 0.0
                    similarity_matrix[j, i] = 0.0
        
        return torch.tensor(similarity_matrix, dtype=torch.float32)
    
    def minmax_normalize(self, matrix):
        
        if isinstance(matrix, torch.Tensor):
            matrix_min = torch.min(matrix)
            matrix_max = torch.max(matrix)
            if matrix_max == matrix_min:
                return torch.ones_like(matrix)
            return (matrix - matrix_min) / (matrix_max - matrix_min)
        else:
            matrix_min = np.min(matrix)
            matrix_max = np.max(matrix)
            if matrix_max == matrix_min:
                return np.ones_like(matrix)
            return (matrix - matrix_min) / (matrix_max - matrix_min)