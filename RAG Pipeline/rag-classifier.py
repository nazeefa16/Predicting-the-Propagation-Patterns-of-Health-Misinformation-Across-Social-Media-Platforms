import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from tqdm.auto import tqdm
import re
import nltk
from sentence_transformers import SentenceTransformer, util
import warnings
import json
import logging
from detailed_evaluation import (
    detailed_model_evaluation,
    compare_models,
    analyze_performance_by_category
)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Try to download NLTK data quietly
try:
    nltk.download('punkt', quiet=True)
except:
    pass

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----------------------------------------
# Data Preprocessing Functions
# ----------------------------------------

def preprocess_text(text):
    """Basic text preprocessing"""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove usernames
        text = re.sub(r'@\w+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    return ""




# ----------------------------------------
# Knowledge Base
# ----------------------------------------

# ----------------------------------------
# General Health Misinformation
# ----------------------------------------
GENERAL_HEALTH_FACTS = {
    "immune_boosting": "The immune system is complex and not easily 'boosted' by single supplements or treatments. A balanced diet, adequate sleep, and regular exercise support overall immune function.",
    "detox_products": "The body has built-in detoxification systems (liver, kidneys) that effectively remove waste and toxins. Commercial 'detox' products are generally unnecessary and lack scientific evidence.",
    "health_conspiracies": "Health conspiracy theories often misrepresent how medical research, healthcare systems, and public health agencies operate, and can lead to harmful health decisions.",
    "natural_always_safe": "Natural products can contain powerful bioactive compounds that may interact with medications or cause side effects; 'natural' does not automatically mean safe or effective.",
    "medical_consensus": "Medical consensus develops through rigorous research, clinical trials, and expert review, not through anecdotes or individual opinions.",
    "health_screening": "Health screening recommendations are based on statistical risk and evidence, not financial incentives for healthcare providers.",
    "radiation_fears": "Normal exposure to Wi-Fi, cell phones, and microwave ovens does not cause significant health risks; these emit non-ionizing radiation that doesn't damage DNA.",
    "chemical_fears": "Not all chemicals are harmful; the dose makes the poison, and many 'chemicals' are essential for life and health.",
    "correlation_causation": "Correlation between two health factors doesn't prove one causes the other; rigorous studies control for multiple variables to establish causation.",
    "medical_research": "Medical research requires multiple studies, peer review, and replication before findings are accepted, not just single preliminary studies."
}

# ----------------------------------------
# Vaccine Misinformation (Beyond COVID)
# ----------------------------------------
GENERAL_VACCINE_FACTS = {
    "autism_vaccines": "Multiple large, well-designed studies have found no link between vaccines and autism. The original study suggesting this link was retracted due to serious procedural and ethical flaws.",
    "vaccine_ingredients": "Vaccine ingredients are carefully tested for safety and used in tiny amounts. Substances like formaldehyde and aluminum are present in lower amounts than occur naturally in the body.",
    "vaccine_schedule": "The recommended vaccine schedule is designed to protect children when they're most vulnerable to diseases, not to overwhelm their immune system.",
    "mercury_vaccines": "Most vaccines no longer contain thimerosal (a mercury-based preservative). When used, the type of mercury (ethylmercury) is quickly eliminated from the body, unlike environmental mercury.",
    "natural_immunity": "Vaccines enable immunity without suffering through potentially dangerous diseases. Vaccine-induced immunity is safer than disease-acquired immunity.",
    "hpv_vaccine": "HPV vaccines prevent infections that can lead to several types of cancer. They do not encourage sexual activity and have strong safety profiles.",
    "flu_vaccine_myths": "The flu vaccine cannot give you the flu as it contains inactivated virus or specific proteins, not live virus. Yearly vaccination is needed because flu viruses constantly change.",
    "infant_immune_system": "Infants' immune systems are capable of responding to numerous vaccines and environmental challenges simultaneously.",
    "vaccine_testing": "Vaccines undergo extensive clinical trials monitoring safety and efficacy before approval, followed by continued safety surveillance after release.",
    "herd_immunity": "High vaccination rates protect vulnerable individuals who cannot be vaccinated through 'herd immunity' or 'community protection.'"
}

# ----------------------------------------
# Nutrition and Diet Misinformation
# ----------------------------------------
NUTRITION_FACTS = {
    "superfoods": "The term 'superfood' is primarily a marketing concept. While certain foods are nutrient-dense, no single food provides all necessary nutrients or prevents disease alone.",
    "detox_diets": "Short-term 'detox' diets or cleanses are unnecessary as the liver and kidneys continuously remove waste products. These diets may lack essential nutrients.",
    "organic_nutrition": "Organic foods may reduce pesticide exposure but have not been proven to be significantly more nutritious than conventionally grown foods.",
    "gmo_safety": "Genetically modified foods currently on the market have passed safety assessments and have not been shown to pose specific health risks to humans.",
    "alkaline_diet": "The body tightly regulates its pH level regardless of diet. Foods cannot significantly change the body's pH, and 'alkaline diets' lack scientific support.",
    "gluten_sensitivity": "Celiac disease requires strict gluten avoidance, but non-celiac 'gluten sensitivity' is controversial and less well-defined medically.",
    "sugar_addiction": "While sugar activates reward pathways in the brain, the concept of sugar 'addiction' matching drug addiction criteria is not fully supported by research.",
    "multivitamin_necessity": "Most people who eat a varied diet don't need multivitamins. Targeted supplementation may be useful for specific deficiencies or life stages.",
    "weight_loss_methods": "Sustainable weight management comes from long-term dietary and lifestyle changes, not 'quick fixes' or extreme diets that are difficult to maintain.",
    "artificial_sweeteners": "Approved artificial sweeteners are thoroughly tested for safety and most people can consume them in moderation without health effects.",
    "clean_eating": "'Clean eating' lacks a scientific definition. Focusing on whole foods is beneficial, but labeling foods as 'clean' vs. 'dirty' can promote unhealthy relationships with food.",
    "dietary_fat": "Not all fats are unhealthy; unsaturated fats from sources like olive oil, nuts, and fish are important for health. Even saturated fats have complex health effects.",
    "carbohydrates": "Carbohydrates are not inherently fattening or unhealthy. Quality (whole vs. refined) and quantity matter more than eliminating this major nutrient group."
}

# ----------------------------------------
# Medication and Treatment Misinformation
# ----------------------------------------
MEDICATION_FACTS = {
    "antibiotics_viruses": "Antibiotics only work against bacterial infections, not viral infections like colds and flu. Misuse contributes to antibiotic resistance.",
    "medication_natural": "Many medications are derived from or inspired by natural compounds, but are purified and standardized for safety and efficacy.",
    "generic_drugs": "Generic medications contain the same active ingredients as brand-name versions and must meet the same quality and efficacy standards.",
    "pain_medication": "When used as directed, pain medications are generally safe and effective. Untreated pain can have serious physical and psychological consequences.",
    "medication_dependency": "Physical dependence on medication is not the same as addiction; many conditions require ongoing medication for management.",
    "placebo_effect": "The placebo effect is real but limited in scope. Placebos generally don't cure diseases or affect objective measures like blood tests.",
    "drug_side_effects": "All medications have potential side effects, but regulatory approval means benefits are judged to outweigh risks for the intended population.",
    "expired_medications": "Many medications remain effective after expiration dates, though potency may gradually decrease. Some medications, particularly liquids, can degrade more quickly.",
    "medication_interactions": "Interactions between medications, supplements, and foods can be serious. Always disclose all substances to healthcare providers.",
    "medication_adherence": "Taking medications as prescribed is crucial for treatment success; stopping early, even when feeling better, can lead to treatment failure or disease recurrence."
}

# ----------------------------------------
# Alternative Medicine Misinformation
# ----------------------------------------
ALTERNATIVE_MEDICINE_FACTS = {
    "homeopathy": "Homeopathic preparations are typically diluted to the point where no molecules of the original substance remain. Scientific evidence does not support effectiveness beyond placebo.",
    "acupuncture": "Acupuncture may help with pain and nausea in some cases, but evidence is mixed. Many claimed benefits lack scientific support.",
    "chiropractic": "While spinal manipulation may help some types of back pain, claims that chiropractic adjustments cure diseases or improve general health lack scientific support.",
    "essential_oils": "Essential oils may have pleasant aromas and some topical uses, but claims about treating diseases or replacing conventional medicine are not supported by research.",
    "energy_healing": "Practices claiming to manipulate invisible 'energy fields' (reiki, therapeutic touch) have not been shown to affect health outcomes beyond relaxation responses.",
    "naturopathy": "Naturopathic practices vary widely in scientific support. Some recommendations align with conventional advice (diet, exercise), while others lack evidence.",
    "traditional_medicine": "Traditional medicine systems may contain valuable treatments, but individual remedies should be evaluated scientifically for safety and efficacy.",
    "supplement_regulation": "Dietary supplements are less strictly regulated than pharmaceuticals and don't require proof of efficacy before marketing.",
    "alternative_cancer": "Alternative cancer treatments used instead of conventional treatment can lead to delayed care and worse outcomes.",
    "chelation": "Chelation therapy is effective for heavy metal poisoning but lacks evidence for treating autism, heart disease, or other conditions for which it's sometimes promoted.",
    "colloidal_silver": "Colloidal silver has no proven health benefits and can cause serious side effects, including permanent bluish skin discoloration (argyria)."
}

# ----------------------------------------
# Mental Health Misinformation
# ----------------------------------------
MENTAL_HEALTH_FACTS = {
    "mental_illness_real": "Mental health conditions are real medical conditions with biological and environmental components, not character flaws or signs of weakness.",
    "depression_treatment": "Depression is a medical condition that often requires professional treatment, not just 'positive thinking' or 'trying harder.'",
    "antidepressants": "Antidepressants don't create artificial happiness but help restore normal brain chemistry. They're not addictive but should be tapered under medical supervision.",
    "adhd_reality": "ADHD is a well-documented neurodevelopmental condition, not just 'bad behavior' or lack of discipline.",
    "therapy_effectiveness": "Psychotherapy is an evidence-based treatment for many mental health conditions, not just 'paying to talk about problems.'",
    "suicide_prevention": "Asking someone directly about suicidal thoughts doesn't increase risk and can be a crucial step in getting help.",
    "addiction_choice": "Addiction involves brain changes affecting behavior and decision-making; it's not simply a choice or moral failing.",
    "anxiety_disorders": "Anxiety disorders are more than normal worry or stress; they involve excessive anxiety that interferes with daily functioning.",
    "mental_health_violence": "Mental health conditions alone rarely cause violent behavior. People with mental illness are more likely to be victims than perpetrators of violence.",
    "ocd_misconceptions": "OCD is a serious anxiety disorder involving intrusive thoughts and repetitive behaviors, not just being neat or organized.",
    "bipolar_disorder": "Bipolar disorder involves distinct episodes of mania and depression, not just frequent mood swings."
}

# ----------------------------------------
# Reproductive Health Misinformation
# ----------------------------------------
REPRODUCTIVE_HEALTH_FACTS = {
    "birth_control": "Hormonal birth control methods work primarily by preventing ovulation; they don't cause abortions or harm future fertility.",
    "fertility_tracking": "Fertility awareness methods require consistent tracking and have higher failure rates than many other contraceptive methods.",
    "sti_protection": "Condoms, when used correctly and consistently, significantly reduce but don't eliminate STI transmission risk.",
    "sex_education": "Comprehensive sex education is associated with later sexual debut and increased contraceptive use, not increased sexual activity.",
    "abortion_safety": "Legal abortion performed by qualified providers is a safe medical procedure with low complication rates.",
    "pregnancy_myths": "Activities like exercise, spicy food, or sex do not typically induce labor unless the body is already preparing for delivery.",
    "infertility_causes": "Infertility affects men and women equally, with male factors contributing to about half of all cases.",
    "ectopic_pregnancy": "Ectopic pregnancies (outside the uterus) are never viable and can be life-threatening if untreated.",
    "emergency_contraception": "Emergency contraception primarily works by preventing ovulation, not by preventing implantation of a fertilized egg.",
    "assisted_reproduction": "IVF and other assisted reproductive technologies result in healthy babies with birth defect rates similar to natural conception."
}

# ----------------------------------------
# Cancer Misinformation
# ----------------------------------------
CANCER_FACTS = {
    "cancer_causes": "Cancer develops from a combination of genetic and environmental factors, not from a single cause like emotional stress or specific foods.",
    "cancer_prevention": "While no lifestyle guarantees cancer prevention, maintaining healthy weight, avoiding tobacco, limiting alcohol, and getting screenings reduce risk.",
    "cancer_treatment": "Standard cancer treatments (surgery, radiation, chemotherapy) are based on extensive research and significantly improve survival rates.",
    "alternative_cancer_treatment": "Alternative treatments used instead of conventional cancer therapy typically lack scientific support and can delay effective treatment.",
    "cancer_screening": "Cancer screening recommendations balance benefits of early detection with risks of overdiagnosis and false positives.",
    "cancer_sugar": "While cancer cells use glucose (sugar) for energy, eliminating sugar from the diet doesn't 'starve' cancer; all cells need glucose.",
    "cancer_spread": "Cancer doesn't spread or accelerate when exposed to air during surgery; this is a myth without scientific basis.",
    "cancer_biopsy": "Biopsies don't cause cancer to spread; this misconception may lead to harmful delays in diagnosis.",
    "cancer_genetics": "While some cancers have genetic components, most are not directly inherited and result from acquired mutations.",
    "cancer_alkaline": "The body's pH is tightly regulated, and diet cannot significantly alter the pH of blood or influence cancer development.",
    "cancer_cures": "Claims of 'hidden' or 'suppressed' cancer cures misrepresent how medical research and cancer treatment development work.",
    "artificial_sweeteners_cancer": "Major scientific and regulatory bodies have found no convincing evidence that approved artificial sweeteners cause cancer in humans."
}

# ----------------------------------------
# Chronic Disease Misinformation
# ----------------------------------------
CHRONIC_DISEASE_FACTS = {
    "diabetes_diet": "Type 2 diabetes requires individualized dietary approaches, not simply avoiding sugar. Carbohydrate quality and quantity, portion control, and overall diet pattern matter.",
    "diabetes_causes": "Type 1 diabetes is an autoimmune condition not caused by diet or lifestyle. Type 2 diabetes involves genetic factors and lifestyle influences.",
    "diabetes_cures": "While Type 2 diabetes can be managed and sometimes put into remission with lifestyle changes, there's no proven 'cure' for either type of diabetes.",
    "heart_disease_prevention": "Heart disease prevention involves multiple approaches including diet, exercise, not smoking, and sometimes medication, not single 'miracle' foods or supplements.",
    "hypertension_causes": "Hypertension (high blood pressure) usually develops from a combination of genetic and lifestyle factors, not primarily from stress or specific foods.",
    "arthritis_treatments": "Arthritis management typically requires medical approaches; many heavily marketed supplements lack strong evidence for effectiveness.",
    "autoimmune_triggers": "Autoimmune disease triggers are complex and not fully understood; simple dietary changes alone rarely 'cure' these conditions.",
    "food_allergy_testing": "Many commercial food 'sensitivity' tests lack scientific validation. True food allergies require specific diagnostic testing by medical professionals.",
    "chronic_fatigue": "Chronic fatigue syndrome/myalgic encephalomyelitis is a real medical condition, not laziness or depression, requiring proper medical care.",
    "fibromyalgia": "Fibromyalgia is a legitimate chronic pain condition, not imaginary or 'all in one's head,' though its mechanisms are still being researched."
}

# ----------------------------------------
# Public Health Misinformation
# ----------------------------------------
PUBLIC_HEALTH_FACTS = {
    "herd_immunity": "Herd immunity protects vulnerable populations when a high percentage of the community is immune to an infectious disease, primarily achieved through vaccination.",
    "fluoride_safety": "Water fluoridation at recommended levels is safe and effective for preventing tooth decay, despite misconceptions about health risks.",
    "vaccine_development": "Vaccine development follows rigorous scientific processes including preclinical testing, multiple phases of clinical trials, and ongoing safety monitoring.",
    "outbreak_response": "Public health measures during disease outbreaks are based on scientific evidence about transmission patterns, not political or economic control.",
    "health_disparities": "Health disparities among racial and socioeconomic groups result from complex social factors, not biological differences or personal choices alone.",
    "mask_effectiveness": "Masks can help reduce respiratory disease transmission by blocking exhaled droplets, especially when widely used in combination with other measures.",
    "handwashing": "Proper handwashing with soap and water is highly effective at reducing germ transmission and preventing infectious diseases.",
    "antibiotic_resistance": "Antibiotic resistance is accelerated by misuse and overuse of antibiotics, threatening the effectiveness of these critical medications.",
    "epidemiological_models": "Disease modeling uses scientific methods to predict outcomes under different scenarios, not to exaggerate threats or control populations.",
    "contact_tracing": "Contact tracing is a standard, evidence-based public health tool for controlling infectious disease outbreaks, not surveillance for other purposes."
}

# ----------------------------------------
# Combine All Knowledge Categories
# ----------------------------------------
KNOWLEDGE_BASE = {
    **GENERAL_HEALTH_FACTS,
    **GENERAL_VACCINE_FACTS,
    **NUTRITION_FACTS,
    **MEDICATION_FACTS,
    **ALTERNATIVE_MEDICINE_FACTS,
    **MENTAL_HEALTH_FACTS,
    **REPRODUCTIVE_HEALTH_FACTS,
    **CANCER_FACTS,
    **CHRONIC_DISEASE_FACTS,
    **PUBLIC_HEALTH_FACTS
}
# ----------------------------------------
# Retrieval Functions
# ----------------------------------------

class BatchRetriever:
    """Optimized retriever with batch processing support"""
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        try:
            self.encoder = SentenceTransformer(model_name)
            self.encoder.to(device)
            self.knowledge_base = None
            self.knowledge_texts = None
            self.knowledge_embeddings = None
            self.knowledge_keys = None
        except Exception as e:
            logger.error(f"Error initializing retriever: {e}")
            raise
    
    def index_knowledge_base(self, knowledge_base):
        """Index the knowledge base with embeddings"""
        try:
            self.knowledge_base = knowledge_base
            self.knowledge_texts = list(knowledge_base.values())
            self.knowledge_keys = list(knowledge_base.keys())
            
            # Compute embeddings
            logger.info("Indexing knowledge base...")
            self.knowledge_embeddings = self.encoder.encode(
                self.knowledge_texts, 
                convert_to_tensor=True,
                show_progress_bar=True
            ).to(device)
            logger.info(f"Indexed {len(self.knowledge_texts)} knowledge items")
        except Exception as e:
            logger.error(f"Error indexing knowledge base: {e}")
            raise
    
    def batch_retrieve(self, queries, top_k=2, batch_size=64):
        """Retrieve knowledge for multiple queries in batches"""
        try:
            all_results = []
            
            # Process in batches for memory efficiency
            for i in range(0, len(queries), batch_size):
                # Get batch of queries
                batch_queries = queries[i:i+batch_size]
                
                # Encode batch
                batch_embeddings = self.encoder.encode(
                    batch_queries,
                    convert_to_tensor=True,
                    show_progress_bar=False
                ).to(device)
                
                # Calculate similarities for batch
                batch_similarities = util.pytorch_cos_sim(batch_embeddings, self.knowledge_embeddings)
                
                # Process each query in batch
                batch_results = []
                for j, similarities in enumerate(batch_similarities):
                    # Get top-k indices
                    top_indices = torch.topk(similarities, min(top_k, len(similarities))).indices.tolist()
                    
                    # Get items and scores
                    items = [self.knowledge_texts[idx] for idx in top_indices]
                    scores = [similarities[idx].item() for idx in top_indices]
                    keys = [self.knowledge_keys[idx] for idx in top_indices]
                    
                    # Store results for this query
                    query_results = list(zip(items, scores, keys))
                    batch_results.append(query_results)
                
                # Add batch results
                all_results.extend(batch_results)
            
            return all_results
        except Exception as e:
            logger.error(f"Error in batch_retrieve: {e}")
            # Return fallback results
            return [[("Health information should be verified with trusted sources.", 0.5, "general")] 
                   for _ in range(len(queries))]
    
    def retrieve(self, query, top_k=2):
        """Single query retrieval for compatibility"""
        try:
            # Use batch retrieval with a single query
            results = self.batch_retrieve([query], top_k=top_k)[0]
            return results
        except Exception as e:
            logger.error(f"Error in retrieve: {e}")
            # Return fallback
            return [("Health information should be verified with trusted sources.", 0.5, "general")]

def process_dataset_optimized(df, retriever, batch_size=128, max_samples=None):
    """Process dataset with optimized batch processing"""
    try:
        logger.info(f"Processing dataset with {len(df)} samples")
        
        # Limit sample size if specified
        if max_samples and len(df) > max_samples:
            df = df.sample(max_samples, random_state=RANDOM_SEED)
            logger.info(f"Limited to {max_samples} samples")
        
        # Extract all texts
        texts = df['processed_text'].tolist()
        
        # Filter out very short texts
        valid_indices = [i for i, text in enumerate(texts) if isinstance(text, str) and len(text) >= 10]
        valid_texts = [texts[i] for i in valid_indices]
        
        logger.info(f"Processing {len(valid_texts)} valid texts in batches of {batch_size}")
        
        # Create results storage
        processed_data = []
        
        # Process in manageable chunks to avoid memory issues
        chunk_size = 1000  # Process 1000 samples at a time
        
        for chunk_start in range(0, len(valid_texts), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(valid_texts))
            chunk_indices = valid_indices[chunk_start:chunk_end]
            chunk_texts = valid_texts[chunk_start:chunk_end]
            
            logger.info(f"Processing chunk {chunk_start//chunk_size + 1} with {len(chunk_texts)} texts")
            
            # Batch retrieve knowledge for chunk
            all_retrieval_results = []
            
            for i in tqdm(range(0, len(chunk_texts), batch_size), desc=f"Retrieving knowledge (chunk {chunk_start//chunk_size + 1})"):
                batch_texts = chunk_texts[i:i+batch_size]
                batch_results = retriever.batch_retrieve(batch_texts, top_k=2, batch_size=batch_size)
                all_retrieval_results.extend(batch_results)
            
            # Process results for this chunk
            for i, idx in enumerate(chunk_indices):
                row = df.iloc[idx]
                text = row['processed_text']
                retrieval_results = all_retrieval_results[i]
                
                # Format knowledge items
                knowledge_items = [f"FACT: {item}" for item, _, _ in retrieval_results]
                knowledge_scores = [score for _, score, _ in retrieval_results]
                
                # Create combined text
                combined_text = combine_text_with_knowledge(text, knowledge_items)
                
                # Store sample
                processed_data.append({
                    'text': text,
                    'knowledge_items': knowledge_items,
                    'knowledge_scores': knowledge_scores,
                    'combined_text': combined_text,
                    'label': row['label'],
                    'label_encoded': row['label_encoded']
                })
        
        # Convert to DataFrame
        processed_df = pd.DataFrame(processed_data)
        logger.info(f"Processed {len(processed_df)} samples")
        
        return processed_df
    except Exception as e:
        logger.error(f"Error in process_dataset_optimized: {e}")
        raise

def combine_text_with_knowledge(text, knowledge_items):
    """Combine text with knowledge"""
    try:
        if not knowledge_items:
            return text
        
        # Create combined text
        combined = f"Text: {text} [SEP] "
        
        # Add knowledge items
        for i, knowledge in enumerate(knowledge_items):
            combined += f"Knowledge {i+1}: {knowledge} [SEP] "
        
        return combined.strip()
    except Exception as e:
        logger.error(f"Error combining text with knowledge: {e}")
        return text

# ----------------------------------------
# Model Definition
# ----------------------------------------

class SimpleRAGModel(nn.Module):
    """Simple RAG classification model"""
    def __init__(self, base_model_name, num_labels=2):
        super().__init__()
        # Load pre-trained model
        self.base_model = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.base_model.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )
        
        # Set config for compatibility with HF Trainer
        self.config = self.base_model.config
        self.config.num_labels = num_labels
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get CLS token representation
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # Classification
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            
        return (loss, logits) if loss is not None else logits

# ----------------------------------------
# Data Processing
# ----------------------------------------

def prepare_datasets(train_df, val_df, test_df, tokenizer, text_column="combined_text", max_length=256):
    """Prepare datasets for training"""
    try:
        logger.info("Preparing datasets for training")
        
        # Tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples[text_column],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Convert to Datasets
        train_dataset = Dataset.from_pandas(train_df[[text_column, 'label_encoded']])
        val_dataset = Dataset.from_pandas(val_df[[text_column, 'label_encoded']])
        test_dataset = Dataset.from_pandas(test_df[[text_column, 'label_encoded']])
        
        # Tokenize with batching
        tokenized_train = train_dataset.map(
            lambda x: {**tokenize_function(x), "labels": x["label_encoded"]},
            batched=True,
            batch_size=128,
            desc="Tokenizing train"
        )
        
        tokenized_val = val_dataset.map(
            lambda x: {**tokenize_function(x), "labels": x["label_encoded"]},
            batched=True,
            batch_size=128,
            desc="Tokenizing validation"
        )
        
        tokenized_test = test_dataset.map(
            lambda x: {**tokenize_function(x), "labels": x["label_encoded"]},
            batched=True,
            batch_size=128,
            desc="Tokenizing test"
        )
        
        # Set format for PyTorch
        tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        tokenized_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        return tokenized_train, tokenized_val, tokenized_test
    except Exception as e:
        logger.error(f"Error preparing datasets: {e}")
        raise

# ----------------------------------------
# Training and Evaluation
# ----------------------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def perform_detailed_evaluation(rag_model, rag_test, tokenizer, baseline_model, baseline_test, test_df):
    """
    Run detailed evaluation of both models and compare results.
    """
    # Make sure output directory exists
    os.makedirs("./output/evaluation", exist_ok=True)
    
    logger.info("Performing detailed evaluation")
    
    # Convert test_df to have the right format
    test_data = test_df.copy()
    
    # Try to make test_dataset objects have the text column available for error analysis
    try:
        # Add text field to test datasets if not already present
        if hasattr(rag_test, 'orig_dataset'):
            rag_test.orig_dataset = Dataset.from_pandas(test_data[['text', 'knowledge_items', 'label_encoded']])
        
        if hasattr(baseline_test, 'orig_dataset'):
            baseline_test.orig_dataset = Dataset.from_pandas(test_data[['text', 'label_encoded']])
    except Exception as e:
        logger.warning(f"Could not add text fields to test datasets: {e}")
    
    # Run detailed evaluation on RAG model
    rag_results = detailed_model_evaluation(
        rag_model,                 
        rag_test,                  
        output_dir="./output/evaluation",
        model_name="rag_model",
        threshold=0.5,            
        knowledge_column="knowledge_items",   
        text_column="text"        
    )
    
    # Run detailed evaluation on baseline model
    baseline_results = detailed_model_evaluation(
        baseline_model,           
        baseline_test,            
        output_dir="./output/evaluation",
        model_name="baseline_model",
        threshold=0.5,            
        text_column="text_only"   
    )
    
    # Compare models
    comparison = compare_models(
        baseline_results=baseline_results,
        rag_results=rag_results,
        output_dir="./output/evaluation"
    )
    
    # Extract required data for category analysis
    test_texts = test_data['text'].tolist()
    test_labels = test_data['label_encoded'].tolist()
    
    # Get predictions for both models
    trainer_baseline = Trainer(model=baseline_model)
    trainer_rag = Trainer(model=rag_model)
    
    baseline_output = trainer_baseline.predict(baseline_test)
    rag_output = trainer_rag.predict(rag_test)
    
    baseline_probs = torch.nn.functional.softmax(torch.tensor(baseline_output.predictions), dim=1).numpy()
    rag_probs = torch.nn.functional.softmax(torch.tensor(rag_output.predictions), dim=1).numpy()
    
    baseline_preds = (baseline_probs[:, 1] >= 0.5).astype(int)
    rag_preds = (rag_probs[:, 1] >= 0.5).astype(int)
    
    # Run category analysis
    category_results = analyze_performance_by_category(
        texts=test_texts,
        true_labels=test_labels,
        baseline_preds=baseline_preds,
        rag_preds=rag_preds,
        output_dir="./output/evaluation"
    )
    
    # Log summary of findings
    logger.info("Detailed evaluation completed")
    logger.info(f"Baseline model overall accuracy: {baseline_results['accuracy']:.4f}")
    logger.info(f"RAG model overall accuracy: {rag_results['accuracy']:.4f}")
    
    # Identify strengths/weaknesses of RAG model
    better_categories = category_results[category_results['accuracy_diff'] > 0.02]['category'].tolist()
    worse_categories = category_results[category_results['accuracy_diff'] < -0.02]['category'].tolist()
    
    if better_categories:
        logger.info(f"RAG model performs better on: {', '.join(better_categories)}")
    
    if worse_categories:
        logger.info(f"RAG model performs worse on: {', '.join(worse_categories)}")
    
    # Analysis of error types
    fp_diff = rag_results['fp'] - baseline_results['fp']
    fn_diff = rag_results['fn'] - baseline_results['fn']
    
    logger.info(f"RAG model false positives: {rag_results['fp']} ({fp_diff:+d} vs baseline)")
    logger.info(f"RAG model false negatives: {rag_results['fn']} ({fn_diff:+d} vs baseline)")
    
    return {
        "baseline": baseline_results,
        "rag": rag_results,
        "comparison": comparison,
        "category_analysis": category_results
    }



def train_and_evaluate(
    model, 
    train_dataset, 
    val_dataset, 
    test_dataset, 
    output_dir,
    num_epochs=3,
    batch_size=128,
    gradient_accumulation_steps=1
):
    """Train and evaluate model with optimized settings"""
    try:
        logger.info(f"Training model for {num_epochs} epochs with batch size {batch_size}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=1,
        warmup_ratio=0.01,
        weight_decay=0.03,
        learning_rate=3e-5,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",  
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_drop_last=False,
        remove_unused_columns=False
    )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train model
        logger.info("Starting training")
        trainer.train()
        
        # Evaluate on test set
        logger.info("Evaluating on test set")
        test_results = trainer.evaluate(test_dataset)
        logger.info(f"Test results: {test_results}")
        
        # Save model
        trainer.save_model(f"{output_dir}/final_model")
        
        # Save results
        with open(f"{output_dir}/test_results.json", 'w') as f:
            json.dump(test_results, f, indent=2)
        
        return test_results
    except Exception as e:
        logger.error(f"Error in train_and_evaluate: {e}")
        # Return empty results
        return {
            'eval_loss': 0.0,
            'eval_accuracy': 0.0,
            'eval_f1': 0.0,
            'eval_precision': 0.0,
            'eval_recall': 0.0
        }

# ----------------------------------------
# Main Function
# ----------------------------------------

def main():
    try:
        # Check environment
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Transformers version: {transformers.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        # Load dataset
        logger.info("Loading dataset")
        try:
            # Load the actual dataset
            dataset_path = '/home/aledel/repos/Predicting-the-Propagation-Patterns-of-Health-Misinformation-Across-Social-Media-Platforms/merged_dataset.csv'
            df = pd.read_csv(dataset_path)
            logger.info(f"Loaded dataset with {len(df)} entries")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise  # Stop execution if dataset can't be loaded
        
        # Preprocess data
        logger.info("Preprocessing data")
        df['processed_text'] = df['content'].apply(preprocess_text)
        df['label_encoded'] = df['label'].apply(lambda x: 1 if x.lower() == 'misinformation' else 0)
        
        
        # Initialize retriever with optimized batch processing
        logger.info("Initializing batch retriever")
        retriever = BatchRetriever()
        retriever.index_knowledge_base(KNOWLEDGE_BASE)
        
        # Process dataset with RAG in optimized batches
        logger.info("Processing dataset with optimized batch retrieval")
        processed_data = process_dataset_optimized(
            df, 
            retriever,
            batch_size=128
        )
        
        # Split data
        logger.info("Splitting data")        
        
        train_df, temp_df = train_test_split(
            processed_data, 
            test_size=0.2,
            random_state=RANDOM_SEED,
            stratify=processed_data['label_encoded']
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=RANDOM_SEED,
            stratify=temp_df['label_encoded']
        )
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Calculate class weights
        label_counts = train_df['label_encoded'].value_counts()
        total = len(train_df)
        num_pos = label_counts.get(1, 0)
        num_neg = label_counts.get(0, 0)

        # Compute class weights for the loss function
        weight_for_0 = (1 / num_neg) * (total / 2) if num_neg > 0 else 1.0
        weight_for_1 = (1 / num_pos) * (total / 2) if num_pos > 0 else 1.0

        # Adjust weight_for_1 to prioritize recall over precision
        # Higher value = higher recall but potentially lower precision
        weight_for_1 *= 1.2  # This simple adjustment favors recall

        class_weights = {0: weight_for_0, 1: weight_for_1}
        print(f"Using class weights: {class_weights}")


        # Prepare text-only baseline
        train_df_baseline = train_df.copy()
        val_df_baseline = val_df.copy()
        test_df_baseline = test_df.copy()
        
        train_df_baseline['text_only'] = train_df_baseline['text']
        val_df_baseline['text_only'] = val_df_baseline['text']
        test_df_baseline['text_only'] = test_df_baseline['text']
        
        # Initialize tokenizer
        logger.info("Initializing tokenizer")
        model_name = "distilbert/distilroberta-base"  
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Prepare datasets with batch processing
        logger.info("Preparing datasets")
        # RAG datasets
        rag_train, rag_val, rag_test = prepare_datasets(
            train_df, val_df, test_df,
            tokenizer, 
            text_column="combined_text"
        )
        
        # Baseline datasets
        baseline_train, baseline_val, baseline_test = prepare_datasets(
            train_df_baseline, val_df_baseline, test_df_baseline,
            tokenizer, 
            text_column="text_only"
        )
        
        # Determine optimal batch size based on dataset size and hardware
        is_large_dataset = len(train_df) > 1000
        has_gpu = torch.cuda.is_available()
        
        if is_large_dataset and has_gpu:
            batch_size = 128
            grad_accum = 1
        elif is_large_dataset:
            batch_size = 32
            grad_accum = 4
        elif has_gpu:
            batch_size = 64
            grad_accum = 1
        else:
            batch_size = 16
            grad_accum = 2
            
        logger.info(f"Using batch size {batch_size} with gradient accumulation {grad_accum}")
        
        # Train RAG model
        logger.info("Training RAG model")
        rag_model = SimpleRAGModel(model_name)
        rag_model.to(device)
        
        rag_results = train_and_evaluate(
            rag_model,
            rag_train,
            rag_val,
            rag_test,
            output_dir="./output/rag_model",
            num_epochs=3,
            batch_size=batch_size,
            gradient_accumulation_steps=grad_accum
        )
        
        # Train baseline model
        logger.info("Training baseline model")
        baseline_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        baseline_model.to(device)
        
        baseline_results = train_and_evaluate(
            baseline_model,
            baseline_train,
            baseline_val,
            baseline_test,
            output_dir="./output/baseline_model",
            num_epochs=3,
            batch_size=batch_size,
            gradient_accumulation_steps=grad_accum
        )
        
        # Compare results
        logger.info("Comparing results")
        print("\n=== COMPARISON OF RESULTS ===")
        print("Metric       | Baseline | RAG Model | Difference")
        print("-" * 50)
        
        for metric in ['accuracy', 'f1', 'precision', 'recall']:
            baseline_value = baseline_results.get(f'eval_{metric}', 0)
            rag_value = rag_results.get(f'eval_{metric}', 0)
            
            difference = rag_value - baseline_value
            print(f"{metric.ljust(12)}| {baseline_value:.4f} | {rag_value:.4f} | {difference:+.4f}")
        
        logger.info("Running detailed evaluation")

        detailed_results = perform_detailed_evaluation(rag_model, rag_test, baseline_model, baseline_test, test_df)
        # Save final results
        final_results = {
            'baseline': {k: float(v) for k, v in baseline_results.items()},
            'rag': {k: float(v) for k, v in rag_results.items()},
            'dataset_size': len(processed_data),
            'original_dataset_size': len(df)
        }
        
        os.makedirs("./output", exist_ok=True)
        with open('./output/final_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info("Completed successfully")
        return final_results
    
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.exception("Exception details:")
        
        # Try to save error information
        try:
            os.makedirs("./output", exist_ok=True)
            with open('./output/error_log.json', 'w') as f:
                json.dump({
                    'error': str(e),
                    'traceback': str(e.__traceback__)
                }, f, indent=2)
        except:
            pass

if __name__ == "__main__":
    main()