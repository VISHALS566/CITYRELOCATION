import streamlit as st
import streamlit.components.v1 as components 
import pandas as pd
import sqlite3
import os
import random
import json
import requests
import re
import plotly.graph_objects as go
from math import radians, cos, sin, asin, sqrt
from dotenv import load_dotenv

load_dotenv()

OLA_API_KEY = os.getenv("OLA_API_KEY")
JSONBIN_KEY = os.getenv("JSONBIN_MASTER_KEY")
LOCALITY_BIN = os.getenv("LOCALITY_BIN_ID")
RESILIENCE_BIN = os.getenv("RESILIENCE_BIN_ID")

@st.cache_data(ttl=3600)
def fetch_jsonbin(bin_id):
    """Fetches data from JSONBin.io."""
    if not bin_id or not JSONBIN_KEY: return {}
    url = f"https://api.jsonbin.io/v3/b/{bin_id}/latest"
    headers = {"X-Master-Key": JSONBIN_KEY}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            return resp.json().get("record", {})
    except: pass
    return {}

def haversine(lat1, lon1, lat2, lon2):
    """Calculates the great circle distance between two points."""
    if any(x == 0 for x in [lat1, lon1, lat2, lon2]): return 0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 6371 * 2 * asin(sqrt(a))

def plot_cost(data):
    """Visualizes the cost breakdown."""
    fig = go.Figure(data=[go.Pie(
        labels=["Rent", "Commute", "Bills", "Savings"], 
        values=[data["Rent"], data["Commute"], data["Overheads"], max(0, data["Remaining"])], 
        hole=.5, 
        marker_colors=['#FF4B4B', '#F1C40F', '#3498DB', '#2ECC71']
    )])
    fig.update_layout(
        margin=dict(t=10, b=20, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False 
    )
    return fig

@st.cache_data(ttl=3600)
def fetch_osrm_data(lat1, lon1, lat2, lon2):
        try:
            url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                d = r.json()
                if "routes" in d and len(d["routes"]) > 0:
                    route = d["routes"][0]
                    geometry = route["geometry"]
                    distance_km = route["distance"] / 1000
                    duration_mins = route["duration"] / 60  
                    return geometry, distance_km, duration_mins
        except: pass
        return None,0,0

try:
    from appp import get_sql_from_llm, get_ai_coordinates
except ImportError:
    st.error("⚠️ 'appp.py' not found. Please ensure appp.py is in the same directory.")
    st.stop()

class ChennaiGeoEngine:
    def __init__(self):
        self.coords_db = fetch_jsonbin(LOCALITY_BIN)
        if not self.coords_db: self.coords_db = {}

    def get_route_geometry(self,lat1,lon1,lat2,lon2):
        geom,distance,duration=fetch_osrm_data(lat1,lon1,lat2,lon2)
        if distance>0:
            return geom,distance,duration
        return None, haversine(lat1, lon1, lat2, lon2),3*haversine(lat1,lon1,lat2,lon2)
    def resolve_workplace(self, query):
        clean = str(query).lower().strip()
        if clean in self.coords_db: 
            val = self.coords_db[clean]
            return (float(val[0]), float(val[1])), "Database"
        try:
            llm_lat_lon = get_ai_coordinates(clean)
            if llm_lat_lon and llm_lat_lon != (0.0, 0.0):
                return llm_lat_lon, "LLM Generated"
        except:
            pass
        return self.resolve_coordinates(query)

    def resolve_coordinates(self, query):
        clean = str(query).lower().strip()
        
        for key in self.coords_db:
            if key in clean:
                val = self.coords_db[key]
                return (float(val[0]), float(val[1])), "Database Partial"
        try:
            url = "https://nominatim.openstreetmap.org/search"
            resp = requests.get(url, params={'q': f"{query}, Chennai", 'format': 'json', 'limit': 1}, headers={'User-Agent': 'ReloApp/1.0'}, timeout=2)
            if resp.status_code == 200 and resp.json():
                d = resp.json()[0]
                return (float(d['lat']), float(d['lon'])), "Online"
        except: pass
        
        return (0.0, 0.0), "Unknown"
    
    

    def generate_map_html_string(self, df, route_geometry=None, work_coords=None, hospitals_list=None):
        map_points = []
        for _, row in df.iterrows():
            if row.get('Latitude', 0) != 0:
                map_points.append({
                    "lat": row['Latitude'], 
                    "lon": row['Longitude'], 
                    "name": row['Locality'], 
                    "rent": f"₹{row['Average_Rent']}"
                })
        
        work_data = []
        if work_coords and work_coords[0] != 0:
            work_data.append({
                "lat": work_coords[0],
                "lon": work_coords[1],
                "popup": "<b>Workplace</b>"
            })

        hosp_data = []
        if hospitals_list:
            for h in hospitals_list:
                time_info = h.get('time', 'N/A')
                hosp_data.append({
                    "lat": h['lat'],
                    "lon": h['lon'],
                    "popup": f"<b>{h['name']}</b><br>{h['dist']} km<br>{time_info}"
                })

        route_json = json.dumps(route_geometry) if route_geometry else "null"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8" />
            <title>Map</title>
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <style>
                body {{ margin: 0; padding: 0; }}
                #map {{ width: 100%; height: 500px; border-radius: 12px; }}
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                // Wait for Leaflet to load fully
                window.onload = function() {{
                    if (typeof L === 'undefined') {{
                        document.getElementById('map').innerHTML = '<div style="padding:20px;text-align:center;color:red">Map failed to load. Please refresh.</div>';
                        return;
                    }}

                    var map = L.map('map').setView([13.05, 80.23], 11);

                    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                        attribution: '&copy; OpenStreetMap &copy; CARTO',
                        maxZoom: 19
                    }}).addTo(map);

                    // Data from Python
                    var properties = {json.dumps(map_points)};
                    var workplace = {json.dumps(work_data)};
                    var hospitals = {json.dumps(hosp_data)};
                    var route = {route_json};

                    var bounds = L.latLngBounds();

                    // Plot Properties
                    properties.forEach(function(p) {{
                        var marker = L.circleMarker([p.lat, p.lon], {{
                            color: '#2ECC71', 
                            radius: 8,
                            fillOpacity: 0.8
                        }}).addTo(map);
                        marker.bindPopup("<b>" + p.name + "</b><br>" + p.rent);
                        bounds.extend([p.lat, p.lon]);
                    }});

                    // Plot Workplace
                    workplace.forEach(function(w) {{
                        var icon = L.icon({{
                            iconUrl: 'https://cdn-icons-png.flaticon.com/512/3005/3005357.png',
                            iconSize: [32, 32],
                            iconAnchor: [16, 32],
                            popupAnchor: [0, -32]
                        }});
                        L.marker([w.lat, w.lon], {{icon: icon}}).addTo(map).bindPopup(w.popup);
                        bounds.extend([w.lat, w.lon]);
                    }});

                    // Plot Hospitals
                    hospitals.forEach(function(h) {{
                        var icon = L.icon({{
                            iconUrl: 'https://cdn-icons-png.flaticon.com/512/4320/4320371.png',
                            iconSize: [24, 24],
                            iconAnchor: [12, 24],
                            popupAnchor: [0, -24]
                        }});
                        L.marker([h.lat, h.lon], {{icon: icon}}).addTo(map).bindPopup(h.popup);
                        bounds.extend([h.lat, h.lon]);
                    }});

                    // Plot Route
                    if (route) {{
                        L.geoJSON(route, {{
                            style: {{ color: '#3388ff', weight: 5, opacity: 0.7 }}
                        }}).addTo(map);
                    }}

                    // Adjust View
                    if (properties.length > 0 || workplace.length > 0) {{
                        map.fitBounds(bounds, {{padding: [50, 50]}});
                    }}
                }};
            </script>
        </body>
        </html>
        """
        return html

geo_engine = ChennaiGeoEngine()


class HospitalEngine:
    def __init__(self):
        self.csv_url= st.secrets.get("HIDDEN_CSV_URL1",None)
        self.data = None
        if self.csv_url:
            try:
                self.data = pd.read_csv(self.csv_url)
                self.data.columns = [c.strip().lower() for c in self.data.columns]
                self.lat_col = next((c for c in self.data.columns if 'lat' in c), None)
                self.lon_col = next((c for c in self.data.columns if 'lon' in c), None)
                self.name_col = next((c for c in self.data.columns if 'hospital' in c or 'name' in c), None)
                self.addr_col = next((c for c in self.data.columns if 'address' in c or 'geo' in c), None)
                self.data = self.data.dropna(subset=[self.lat_col, self.lon_col])
                self.data[self.lat_col] = self.data[self.lat_col].astype(float)
                self.data[self.lon_col] = self.data[self.lon_col].astype(float)
            except: pass
        
        if self.data is None or self.data.empty or not getattr(self, 'lat_col', None):
            self.data = pd.DataFrame({
                'name': ['Apollo', 'MIOT', 'Fortis'], 'address': ['Greams Rd', 'Manapakkam', 'Adyar'],
                'latitude': [13.064, 13.027, 13.006], 'longitude': [80.25, 80.18, 80.257]
            })
            self.lat_col, self.lon_col, self.name_col, self.addr_col = 'latitude', 'longitude', 'name', 'address'

    def find_nearest_n(self, lat, lon, n=3):
        if self.data is None or lat == 0: return []
    
        df_copy = self.data.copy()
        df_copy["approx_km"] = df_copy.apply(lambda r: haversine(lat, lon, r[self.lat_col], r[self.lon_col]), axis=1)
        candidates = df_copy.sort_values("approx_km").head(10) 
    
        results = []

        try:
       
            coords_str = f"{lon},{lat};" + ";".join([f"{r[self.lon_col]},{r[self.lat_col]}" for _, r in candidates.iterrows()])
        
            dest_indices = ";".join([str(i+1) for i in range(len(candidates))])
        
            url = f"http://router.project-osrm.org/table/v1/driving/{coords_str}?sources=0&destinations={dest_indices}&annotations=duration,distance"
        
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                durations = data["durations"][0] 
                distances = data["distances"][0] 
            
                for i, (_, row) in enumerate(candidates.iterrows()):
                    addr = str(row[self.addr_col])[:40] + "..." if self.addr_col else "Chennai"
                
                    if durations[i] is not None:
                        results.append({
                            "name": row[self.name_col],
                            "address": addr,
                            "dist": round(distances[i] / 1000, 2), 
                            "time": f"{int(round(durations[i] / 60, 0))} mins", 
                            "lat": row[self.lat_col],
                            "lon": row[self.lon_col]
                        })
        except Exception as e:
            for _, row in candidates.iterrows():
                addr = str(row[self.addr_col])[:40] + "..." if self.addr_col else "Chennai"
                est_time = int(row["approx_km"] * 3) 
                results.append({
                    "name": row[self.name_col], "address": addr,
                    "dist": round(row["approx_km"], 2), "time": f"{est_time} mins (Est)",
                    "lat": row[self.lat_col], "lon": row[self.lon_col]
                })

        results.sort(key=lambda x: x['dist'])
        return results[:n]

class ResilienceEngine:
    def __init__(self):
        if RESILIENCE_BIN and JSONBIN_KEY:
            self.power_grid_data = fetch_jsonbin(RESILIENCE_BIN)
        else:
            self.power_grid_data = {} 

    def get_power_status(self, locality):
        clean = str(locality).lower().strip()
        for k, v in self.power_grid_data.items():
            if k in clean: return v
        return "Unknown (Assume Moderate)"

class CostOfLivingEngine:
    def calculate_total(self, rent, dist_km, salary, mode="Cab", lifestyle="Standard"):
        daily_cost = {"Bike Taxi": 12, "Auto": 18, "Cab": 25, "Public Transport": 2}.get(mode, 25)
        monthly_commute = int((40 + (dist_km * daily_cost)) * 2 * 22)
        factor = 0.8 if lifestyle == "Frugal" else 1.5 if lifestyle == "Lavish" else 1.0
        overheads = int((3000 + 5000 + 2500 + 8000) * factor + 1000)
        
        total = rent + monthly_commute + overheads
        return {
            "Rent": rent, "Commute": monthly_commute, "Overheads": overheads, "Total": total, "Remaining": salary - total,
            "Verdict": "❌ Not Feasible" if total > salary else "⚠️ Tight" if total > 0.7*salary else "✅ Comfortable",
            "Color": "red" if total > salary else "orange" if total > 0.7*salary else "green",
            "Tips": [f"Switch commute (Save ~{monthly_commute//2})"] if monthly_commute > 10000 else []
        }

if 'geo' not in st.session_state:
    st.session_state['geo'] = geo_engine 
    st.session_state['geo'].hospitals = HospitalEngine()
    st.session_state['geo'].resilience = ResilienceEngine()
    st.session_state['geo'].cost_calc = CostOfLivingEngine()
    try:
        dataset_url=st.secrets.get("HIDDEN_CSV_URL",None)
        if dataset_url:
            df = pd.read_csv(dataset_url)
            df.columns = [c.strip().replace(' ', '_').replace('-', '_') for c in df.columns]
            conn = sqlite3.connect(':memory:', check_same_thread=False)
            df.to_sql('properties', conn, index=False, if_exists='replace')
            st.session_state['conn'] = conn
        else: st.error("CSV Not Found"); st.stop()
    except Exception as e:
        st.error(f"❌ Failed to download main dataset: {e}")
        st.stop()
conn = st.session_state['conn']
geo = st.session_state['geo']


st.set_page_config(page_title="Chennai Relocation AI", layout="wide")
st.title("🏡 Chennai Relocation Intelligence")

if 'search_result' not in st.session_state: st.session_state['search_result'] = None
if 'work_coords' not in st.session_state: st.session_state['work_coords'] = (0,0)

c1, c2, c3 = st.columns([2, 1, 1])
with c1: user_query = st.text_input("Filters:", "2BHK near malls under 25k")
with c2: work_input = st.text_input("Workplace:", placeholder="e.g. Ashok Nagar...")
with c3: salary = st.number_input("Monthly Salary:", value=50000, step=1000)

if st.button("Find Homes 🚀", type="primary", use_container_width=True):
    with st.status("🔍 Searching Intelligence Database...", expanded=True) as status:
        try:
            if work_input:
                w_coords, match_type = geo.resolve_workplace(work_input)
                st.session_state['work_coords'] = w_coords
                if "LLM" in match_type: st.toast(f"🤖 LLM found coords for: {work_input}", icon="📍")
        
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(properties)")
            cols = [c[1] for c in cursor.fetchall()]
        
            sql = get_sql_from_llm(user_query, "\n".join(cols)).replace("```sql", "").replace("```", "").strip()
            st.write("📊 Fetching Property Data...")
            res = pd.read_sql_query(sql, conn)
        
            if not res.empty:
                for i, r in res.iterrows():
                    if r.get('Latitude',0) == 0:
                        lat, lon = geo.resolve_coordinates(r['Locality'])[0]
                        res.at[i, 'Latitude'], res.at[i, 'Longitude'] = lat, lon
            
                if 'Pros' not in res.columns:
                    locs = res['Locality'].tolist()
                    ph = ','.join(['?']*len(locs))
                    full = pd.read_sql_query(f"SELECT Locality, Pros, Cons, Flood_Risk_Label FROM properties WHERE Locality IN ({ph})", conn, params=locs)
                    res = pd.merge(res, full, on='Locality', how='left').loc[:,~res.columns.duplicated()]

            st.session_state['search_result'] = res
        except Exception as e: st.error(f"Search Error: {str(e)}")

if st.session_state['search_result'] is not None and not st.session_state['search_result'].empty:
    res = st.session_state['search_result']
    st.divider()
    st.success(f"Found {len(res)} matches!")
    
    t1, t2, t3 = st.tabs(["Analysis & Risks", "Costs & Financial Health", "Data"])
    
    sel = None
    row = None
    hospitals_list = []
    dist = 0
    route_geom = None
    locality_rows = pd.DataFrame() 

    with t1:
        sel = st.selectbox("Select Home:", res['Locality'].unique())
        if sel:
            locality_rows = res[res['Locality'] == sel]
            
            row = locality_rows.iloc[0]
            lat, lon = row['Latitude'], row['Longitude']
            st.subheader(f"🧐 Analysis of {sel}")
            
            c_pro, c_con = st.columns(2)
            with c_pro:
                st.success("✅ **Pros**")
                # Handle cases where data might be missing
                pros_text = row.get('Pros') if pd.notna(row.get('Pros')) else "No specific data available."
                st.write(pros_text)
                
            with c_con:
                st.error("❌ **Cons**")
                cons_text = row.get('Cons') if pd.notna(row.get('Cons')) else "No specific data available."
                st.write(cons_text)
            
            st.divider()
            
            risk = row.get('Flood_Risk_Label', 'Medium')
            color = "red" if "High" in risk else "orange" if "Medium" in risk else "green"
            st.markdown(f"### 🌊 Flood Risk: :{color}[{risk}]")
            if "High" in risk: st.error("Severe inundation in 2015.")
            elif "Medium" in risk: st.warning("Street logging common.")
            else: st.success("Safe from major floods.")

            if lat != 0:
                if st.session_state['work_coords'][0] != 0:
                    route_geom, dist, duration = geo.get_route_geometry(lat, lon, st.session_state['work_coords'][0], st.session_state['work_coords'][1])
                    st.info(f"🚗 Commute: {dist:.1f} km")
                    st.info(f"⏳ Time Taken ( ETA ):- {duration:.1f} minutes")
                
                hospitals_list = geo.hospitals.find_nearest_n(lat, lon, n=3)
            
            # Map
            map_html = geo.generate_map_html_string(res, route_geom, st.session_state['work_coords'], hospitals_list)
            components.html(map_html, height=500)

    with t2:
        if sel and row is not None:
            c_safe, c_cost = st.columns(2)
            
            with c_safe:
                if hospitals_list:
                    st.subheader("🚑 Nearest Hospitals (Top 3)")
                    for h in hospitals_list:
                        st.markdown(f"**{h['name']}**")
                        time_display = h.get('time', 'N/A')
                        st.caption(f"{h['address']} • {h['dist']} km • {time_display}")
                        st.progress(max(0, min(100, int(100 - h['dist']*10))))
                        st.markdown("---")
                
                st.subheader("⚡ Resilience")
                st.info(f"Power Grid: {geo.resilience.get_power_status(sel)}")

            with c_cost:
                st.subheader("💰 Monthly Financial Health")
                
                bhk_options = {}
                
                for idx, r in locality_rows.iterrows():
                    b_val = int(r.get('BHK', 0))
                    r_price = r.get('Average_Rent', 0)
                    
                    if b_val > 0:
                        bhk_options[f"{b_val} BHK"] = r_price
                
                rent_to_use = 0
                if bhk_options:
                    sorted_options = sorted(list(bhk_options.keys()))
                    selected_bhk = st.radio(
                        "Choose Apartment Type:", 
                        sorted_options, 
                        horizontal=True,
                        key="bhk_selector"
                    )
                    rent_to_use = bhk_options[selected_bhk]
                else:
                    st.warning("Specific BHK data unavailable. Using Average.")
                    rent_to_use = row.get('Average_Rent', 0)

                c_mode, c_style = st.columns(2)
                with c_mode: mode = st.selectbox("Commute Mode", ["Cab", "Auto", "Bike Taxi", "Public Transport"])
                with c_style: style = st.selectbox("Lifestyle", ["Standard", "Frugal", "Lavish"])
                
                calc_dist = dist if dist > 0 else 15
                if dist == 0: st.caption("⚠️ Using 15km default distance (Set Workplace to fix)")
                
                breakdown = geo.cost_calc.calculate_total(rent_to_use, calc_dist, salary, mode, style)
                
                st.markdown(f"### Verdict: :{breakdown['Color']}[{breakdown['Verdict']}]")
                if breakdown['Tips']:
                    with st.expander("💡 Financial Insights", expanded=True):
                        for tip in breakdown['Tips']: st.write(f"- {tip}")

                st.metric("Total Monthly Expense", f"₹{breakdown['Total']:,}", delta=f"Savings: ₹{breakdown['Remaining']:,}")
                
                st.plotly_chart(plot_cost(breakdown), width='stretch')

    with t3: st.dataframe(res)