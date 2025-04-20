"""
Knowledge base for health information facts.

This module contains factual health information that can be used
for retrieval-augmented generation (RAG) to improve misinformation detection.
"""

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
# COVID-19 Misinformation
# ----------------------------------------
COVID_FACTS = {
    "covid_vaccine_development": "COVID-19 vaccines underwent standard safety testing. Their rapid development leveraged prior research on similar coronaviruses and unprecedented global collaboration.",
    "covid_vaccine_safety": "COVID-19 vaccines have been administered to billions of people worldwide with strong safety records and monitoring systems tracking adverse events.",
    "mrna_vaccines": "mRNA vaccines do not alter human DNA. They contain instructions for cells to make a protein that triggers an immune response, then the mRNA quickly degrades.",
    "masks_effectiveness": "Masks reduce transmission of respiratory viruses, including COVID-19, by blocking respiratory droplets, especially in indoor settings with limited ventilation.",
    "covid_treatments": "COVID-19 treatments should be evidence-based and prescribed by healthcare professionals. Using unproven treatments can delay effective care.",
    "natural_immunity": "Both vaccination and prior infection provide immunity, but vaccination offers more consistent protection without the risks of disease.",
    "virus_origin": "Scientific evidence indicates COVID-19 most likely originated from animal-to-human transmission. Investigation into specific origins continues.",
    "pcr_tests": "PCR tests for COVID-19 are highly accurate when properly administered. They detect viral genetic material, not bacteria or other viruses.",
    "comorbidities": "COVID-19 can be serious even for healthy individuals. Comorbidities increase risk, but many deaths involve COVID-19 as a primary cause.",
    "long_covid": "Long COVID is a recognized condition where symptoms persist for weeks or months after initial infection, affecting multiple organ systems."
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
# Combine All Knowledge Categories
# ----------------------------------------
KNOWLEDGE_BASE = {
    **GENERAL_HEALTH_FACTS,
    **GENERAL_VACCINE_FACTS,
    **NUTRITION_FACTS,
    **MEDICATION_FACTS,
    **ALTERNATIVE_MEDICINE_FACTS,
    **MENTAL_HEALTH_FACTS,
    **COVID_FACTS,
    **REPRODUCTIVE_HEALTH_FACTS,
    **CANCER_FACTS
}