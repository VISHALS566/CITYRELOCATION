import sqlite3
import pandas as pd
import re
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0.1
)

def setup_database(csv_file):
    """Loads CSV into an in-memory SQLite database."""
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found. Please ensure the file exists.")
        return None

    df = pd.read_csv(csv_file)
    df.columns = [c.strip().replace(' ', '_').replace('-', '_') for c in df.columns]
    
    conn = sqlite3.connect(':memory:')  
    df.to_sql('properties', conn, index=False, if_exists='replace')
    return conn

def get_ai_coordinates(place_name):
    """
    Asks the LLM for the latitude and longitude of a specific place in Chennai
    using a single string prompt.
    """
    prompt = f"""
    You are a Geospatial Assistant for Chennai, India.
    User will give you a place name (company, college, landmark).
    You must return EXACTLY the Latitude and Longitude in this format:
    lat,lon
    
    Example:
    User: "Tidel Park"
    Output: 12.9892, 80.2483
    
    User: "VIT Chennai"
    Output: 12.8406, 80.1534
    
    If you absolutely do not know, return: 0,0
    DO NOT output any other text. Just the numbers.
    
    User Request: {place_name}
    """
    
    try:
        response = model.invoke(prompt)
        content = response.content.strip()
        
        match = re.search(r"(-?\d+\.\d+),\s*(-?\d+\.\d+)", content)
        if match:
            return (float(match.group(1)), float(match.group(2)))
        return (0.0, 0.0)
    except:
        return (0.0, 0.0)
def get_sql_from_llm(user_query, schema_info):
    """
    Generates SQL query using the LLM with a unified prompt that combines 
    Score-based NLP mapping and Internal Geographic Knowledge.
    """
    prompt = f"""
    You are an AI Data Analyst & SQL Expert for Chennai Real Estate.
    
    ### 1. THE DATABASE
    Table Name: properties
    Schema:
    {schema_info}
    
    ### 2. CRITICAL: HYBRID GEOGRAPHIC KNOWLEDGE (USE INTERNAL KNOWLEDGE)
    The user will ask for amenities (Malls, IT Parks, Metro). Your database 'Pros' column might miss these keywords.
    **You MUST filter by specific Chennai Localities known for these amenities.**
    
    * **IF User asks "Near Malls" / "Shopping":**
        * **Knowledge:** Major malls are in Velachery (Phoenix), Anna Nagar (VR), Royapettah (Express), Vadapalani (Nexus/Forum), Mount Road, Chromepet, Thousand Lights.
        * **SQL Logic:** `WHERE (Locality IN ('Velachery', 'Anna Nagar', 'Vadapalani', 'Royapettah', 'Thousand Lights', 'Chromepet', 'Mount Road') OR Pros LIKE '%Mall%' OR Pros LIKE '%Shopping%')`
        
    * **IF User asks "Near IT Parks" / "Office":**
        * **Knowledge:** OMR, Sholinganallur, Taramani, Siruseri, Porur, Guindy, Navalur, Perungudi.
        * **SQL Logic:** `WHERE (Locality IN ('Sholinganallur', 'Guindy', 'Taramani', 'Navalur', 'Porur', 'Perungudi', 'OMR') OR Pros LIKE '%IT Park%' OR Pros LIKE '%Office%')`

    * **IF User asks "Near Metro":**
        * **Knowledge:** Ashok Nagar, Guindy, Vadapalani, Anna Nagar, Alandur, Saidapet, Koyambedu, T Nagar.
        * **SQL Logic:** `WHERE (Locality IN ('Ashok Nagar', 'Guindy', 'Vadapalani', 'Anna Nagar', 'Saidapet', 'Alandur', 'T Nagar') OR Pros LIKE '%Metro%')`

    ### 3. INTELLIGENT INTENT MAPPING (SCORES & VIBES)
    * **"Vibe" / "Posh" / "Fun" / "Luxury":** * Logic: `ORDER BY Lifestyle_Score DESC` OR `WHERE Lifestyle_Score >= 7`.
    * **"Safe" / "Secure" / "Family":** * Logic: `WHERE Safety_Score >= 7 AND Flood_Risk_Label = 'Low Risk'`.
    * **"Connectivity" / "Commute":** * Logic: `ORDER BY Commute_Score DESC` OR `WHERE Commute_Score >= 7`.
    * **"Medical" / "Hospitals" / "Elderly":** * Logic: `WHERE Healthcare_Score >= 6`.
    * **"Best Area" / "Top Rated":** * Logic: `ORDER BY (Safety_Score + Lifestyle_Score + Commute_Score) DESC`.
    * **"Student Friendly":** * Logic: `Average_Rent < 15000 AND Lifestyle_Score >= 5`.

    ### 4. BUDGET, SPELLING & CONSTRAINTS
    * **"Cheap" / "Budget":** `Average_Rent <= 15000`.
    * **"Luxury":** `Average_Rent >= 40000`.
    * **Spelling:** Automatically fix Chennai locality spellings (e.g., "Velacheri" -> "Velachery", "Solinganallur" -> "Sholinganallur").
    * **Filters:** ALWAYS apply Rent limits (e.g., "under 25k") and BHK filters (e.g., "2BHK") if mentioned.

    ### 5. OUTPUT FORMAT RULES
    * Return **ONLY** the raw SQL query. No markdown.
    * **ALWAYS SELECT:** `Locality, BHK, Average_Rent, Flood_Risk_Label, Lifestyle_Score, Safety_Score, Pros, Cons`.
    * **LIMIT 10** unique Localities unless specified otherwise.

    ### 6. EXAMPLES
    User: "2BHK near malls under 25k"
    Reasoning: "Near malls" -> Filter known mall hubs. "Under 25k" -> Rent filter. "2BHK" -> BHK filter.
    SQL: SELECT Locality, BHK, Average_Rent, Lifestyle_Score, Pros, Cons FROM properties WHERE (Locality IN ('Velachery', 'Anna Nagar', 'Vadapalani', 'Royapettah', 'Chromepet') OR Pros LIKE '%Mall%') AND BHK=2 AND Average_Rent <= 25000 LIMIT 10;

    User: "Safe areas with good vibes"
    Reasoning: "Safe" -> Safety_Score >= 7. "Good vibes" -> Lifestyle_Score >= 7.
    SQL: SELECT Locality, BHK, Average_Rent, Safety_Score, Lifestyle_Score, Flood_Risk_Label, Pros, Cons FROM properties WHERE Safety_Score >= 7 AND Lifestyle_Score >= 7 AND Flood_Risk_Label = 'Low Risk' ORDER BY Lifestyle_Score DESC LIMIT 10;

    ---
    User Query: {user_query}
    """

    try:
        response = model.invoke(prompt)
        raw_output = response.content
        clean_sql = re.sub(r"```sql\n?|```", "", raw_output, flags=re.IGNORECASE).strip()
        return clean_sql
    except Exception as e:
        return f"LLM Error: {e}"

def query_database(conn, user_query):
    if conn is None: return "Database connection failed."
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(properties)")
    columns = cursor.fetchall()
    schema_str = "\n".join([f"- {col[1]} ({col[2]})" for col in columns])
    sql_query = get_sql_from_llm(user_query, schema_str)
    print(f"Generated SQL: {sql_query}")  
    try:
        result_df = pd.read_sql_query(sql_query, conn)
        if result_df.empty:
            return "No matching properties found."
        return result_df
    except Exception as e:
        return f"SQL Execution Error: {e}"

