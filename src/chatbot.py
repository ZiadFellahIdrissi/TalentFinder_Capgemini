import torch

from transformers import BertTokenizer, BertForSequenceClassification

import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

 

 

 

# Load trained model, tokenizer, embeddings and dataframe

 

model = BertForSequenceClassification.from_pretrained('./model_directory')

 

tokenizer = BertTokenizer.from_pretrained('./tokenizer_directory')

 

stacked_embeddings = torch.load('embeddings.pt')

 

df = pd.read_pickle('dataframe.pkl')

 

 

 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

 

model.to(device)

 

model.eval()

 

 

 

def encode_description(description):

 

    """Encode a description using the model."""

 

    inputs = tokenizer(description, return_tensors="pt", max_length=512, truncation=True, padding='max_length').to(device)

 

    with torch.no_grad():

 

        outputs = model.bert(**inputs)

 

    return outputs.last_hidden_state.mean(dim=1)

 

def format_experience(experience):

    years = int(experience)

    months = round((experience - years) * 12)

   

    if years == 0 and months == 0:

        return "No experience"

    elif years == 0:

        return f"{months} month{'s' if months > 1 else ''}"

    elif months == 0:

        return f"{years} year{'s' if years > 1 else ''}"

    else:

        return f"{years} year{'s' if years > 1 else ''} and {months} month{'s' if months > 1 else ''}"

 

def get_similar_profiles(description, top_k=3, threshold= 0.65 ):

 

    """Get top_k profiles most similar to the given description."""

 

   

 

    encoded_desc = encode_description(description)

 

 

 

    similarities = cosine_similarity(encoded_desc.cpu(), stacked_embeddings.cpu())

 

 

 

    # Get top_k indices

 

    top_indices = similarities[0].argsort()[-top_k:][::-1]

 

    similar_profiles = df.iloc[top_indices]

 

    # Calculate similarity scores and add them to the DataFrame

 

    similarity_scores = similarities[0][top_indices]

 

    similar_profiles['similarity_score'] = similarity_scores

 

    if similarity_scores.max() < threshold:

        return []

    # Prepare the data for rendering in results.html

 

    profiles_list = []

 

 

 

    for _, profile in similar_profiles.iterrows():

        phone_number = profile.get("Phone")

        if pd.isna(phone_number) or phone_number is None:

            phone_number = "undefined"

 

        total_experience = profile.get("DurationEmployment", "")

 

        # Convertir la chaîne "X.Y years" en float X.Y

        if isinstance(total_experience, str) and "years" in total_experience:

            total_experience = total_experience.split()[0]

            total_experience = float(total_experience)

 

        profile_data = {

 

            "Full Name": profile.get("Full Name", ""),

 

            "Job": profile.get("Job", ""),

 

            "Profil Url": profile.get("Profil Url", ""),

 

            "Skills": profile.get("Skills", ""),

 

           "Matched by": profile.get("similarity_score", ""),

            "Phone": phone_number,

            "Total Experience" : total_experience,

            "Diploma": profile.get("Diplome", "")

 

           

 

        }

 

 

 

        profiles_list.append(profile_data)

 

 

 

    return profiles_list