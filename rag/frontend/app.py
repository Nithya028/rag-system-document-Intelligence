# import streamlit as st
# import requests
# import json
# import os
# import base64
# from datetime import datetime
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots
# import pandas as pd
# from streamlit_option_menu import option_menu
# from streamlit_chat import message
# import time

# # Configuration
# BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# # Page configuration
# st.set_page_config(
#     page_title="RAG System - Document Intelligence",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Enhanced CSS for modern UI
# st.markdown("""
# <style>
#     .main {
#         padding-top: 1rem;
#     }
    
#     .stAlert {
#         padding: 1rem;
#         margin: 1rem 0;
#         border-radius: 0.5rem;
#     }
    
#     .chat-container {
#         max-height: 600px;
#         overflow-y: auto;
#         padding: 1rem;
#         border: 1px solid #e9ecef;
#         border-radius: 0.75rem;
#         margin: 1rem 0;
#         background: #f8f9fa;
#     }
    
#     .upload-section {
#         border: 2px dashed #4ECDC4;
#         border-radius: 15px;
#         padding: 3rem 2rem;
#         text-align: center;
#         margin: 2rem 0;
#         background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
#         transition: all 0.3s ease;
#     }
    
#     .upload-section:hover {
#         border-color: #45B7D1;
#         background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
#         transform: translateY(-2px);
#     }
    
#     .metric-card {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         padding: 1.5rem;
#         border-radius: 1rem;
#         color: white;
#         text-align: center;
#         box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
#         margin: 0.5rem 0;
#     }
    
#     .insight-card {
#         background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
#         padding: 1rem;
#         border-radius: 0.75rem;
#         color: white;
#         margin: 0.5rem 0;
#         box-shadow: 0 2px 10px rgba(240, 147, 251, 0.3);
#     }
    
#     .viz-container {
#         background: white;
#         border-radius: 1rem;
#         padding: 1rem;
#         box-shadow: 0 4px 20px rgba(0,0,0,0.1);
#         margin: 1rem 0;
#     }
    
#     .stButton > button {
#         border-radius: 25px;
#         border: none;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         font-weight: bold;
#         transition: all 0.3s ease;
#     }
    
#     .stButton > button:hover {
#         background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
#         transform: translateY(-2px);
#         box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
#     }
    
#     .sidebar .stSelectbox {
#         background: white;
#         border-radius: 10px;
#     }
    
#     /* Custom animations */
#     @keyframes fadeIn {
#         from { opacity: 0; transform: translateY(20px); }
#         to { opacity: 1; transform: translateY(0); }
#     }
    
#     .element-container {
#         animation: fadeIn 0.5s ease-in;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "uploaded_files" not in st.session_state:
#     st.session_state.uploaded_files = []
# if "current_documents" not in st.session_state:
#     st.session_state.current_documents = []

# def check_backend_health():
#     """Check if backend is accessible"""
#     try:
#         response = requests.get(f"{BACKEND_URL}/health", timeout=5)
#         return response.status_code == 200
#     except:
#         return False

# def upload_files(files):
#     """Upload files to backend"""
#     try:
#         files_data = []
#         for file in files:
#             files_data.append(("files", (file.name, file.getvalue(), file.type)))
        
#         response = requests.post(f"{BACKEND_URL}/upload", files=files_data)
#         response.raise_for_status()
#         return response.json()
#     except Exception as e:
#         st.error(f"❌ Error uploading files: {str(e)}")
#         return None

# def send_chat_message(query, include_viz=True):
#     """Send chat message to backend"""
#     try:
#         chat_data = {
#             "query": query,
#             "chat_history": [
#                 {"role": msg["role"], "content": msg["content"]} 
#                 for msg in st.session_state.chat_history[-10:]
#             ],
#             "include_visualization": include_viz,
#             "max_results": 5
#         }
        
#         response = requests.post(f"{BACKEND_URL}/chat", json=chat_data)
#         response.raise_for_status()
#         return response.json()
#     except Exception as e:
#         st.error(f"❌ Error sending message: {str(e)}")
#         return None

# def get_analytics(query, chart_types=None):
#     """Get analytics from backend"""
#     try:
#         analytics_data = {
#             "query": query,
#             "chart_types": chart_types or ["bar", "line", "scatter", "histogram"]
#         }
        
#         response = requests.post(f"{BACKEND_URL}/analytics", json=analytics_data)
#         response.raise_for_status()
#         return response.json()
#     except Exception as e:
#         st.error(f"❌ Error getting analytics: {str(e)}")
#         return None

# def get_documents():
#     """Get list of uploaded documents"""
#     try:
#         response = requests.get(f"{BACKEND_URL}/documents")
#         response.raise_for_status()
#         return response.json()["documents"]
#     except Exception as e:
#         st.error(f"❌ Error getting documents: {str(e)}")
#         return []

# def get_visualization(query):
#     """Get visualization from backend"""
#     try:
#         viz_data = {"query": query}
#         response = requests.post(f"{BACKEND_URL}/visualization", json=viz_data)
#         response.raise_for_status()
#         return response.json()
#     except Exception as e:
#         st.error(f"❌ Error generating visualization: {str(e)}")
#         return None

# def delete_document(file_id):
#     """Delete a document"""
#     try:
#         response = requests.delete(f"{BACKEND_URL}/documents/{file_id}")
#         response.raise_for_status()
#         return True
#     except Exception as e:
#         st.error(f"❌ Error deleting document: {str(e)}")
#         return False

# def render_plotly_visualization(viz_data):
#     """Render Plotly visualization with enhanced styling."""
#     if not viz_data:
#         return
    
#     try:
#         if viz_data.get("type") == "plotly":
#             # Create Plotly figure from JSON data
#             fig_data = viz_data.get("data", {})
#             config = viz_data.get("config", {})
            
#             # Enhanced config for better interactivity
#             enhanced_config = {
#                 "displayModeBar": True,
#                 "displaylogo": False,
#                 "modeBarButtonsToRemove": ['pan2d', 'lasso2d', 'select2d', 'autoScale2d'],
#                 "toImageButtonOptions": {
#                     "format": "png",
#                     "filename": "chart",
#                     "height": 600,
#                     "width": 1000,
#                     "scale": 2
#                 },
#                 "responsive": True,
#                 **config
#             }
            
#             # Create the figure
#             fig = go.Figure(fig_data)
            
#             # Display with custom container
#             st.markdown('<div class="viz-container">', unsafe_allow_html=True)
#             st.plotly_chart(fig, use_container_width=True, config=enhanced_config)
#             st.markdown('</div>', unsafe_allow_html=True)
            
#             # Add description
#             if viz_data.get("description"):
#                 st.caption(f"📊 {viz_data['description']}")
                
#         elif viz_data.get("type") == "image":
#             # Handle base64 images (fallback)
#             image_data = viz_data.get("data", "")
#             if image_data:
#                 if image_data.startswith("data:image"):
#                     image_data = image_data.split(",")[1]
                
#                 image_bytes = base64.b64decode(image_data)
#                 st.markdown('<div class="viz-container">', unsafe_allow_html=True)
#                 st.image(image_bytes, caption=viz_data.get("description", "Visualization"), use_column_width=True)
#                 st.markdown('</div>', unsafe_allow_html=True)
#         else:
#             st.warning(f"⚠️ Unknown visualization type: {viz_data.get('type')}")
            
#     except Exception as e:
#         st.error(f"❌ Error rendering visualization: {str(e)}")

# def display_insights(insights):
#     """Display insights with attractive cards."""
#     if insights:
#         st.markdown("### 💡 Key Insights")
        
#         # Display insights in attractive cards
#         for i, insight in enumerate(insights):
#             if i % 2 == 0:
#                 cols = st.columns(2)
            
#             with cols[i % 2]:
#                 st.markdown(
#                     f'<div class="insight-card">{insight}</div>',
#                     unsafe_allow_html=True
#                 )

# def main():
#     # Modern header with gradient
#     st.markdown("""
#         <div style='text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin: -1rem -1rem 2rem -1rem; color: white; border-radius: 0 0 2rem 2rem;'>
#             <h1 style='margin: 0; font-size: 3rem;'>🤖 RAG System</h1>
#             <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;'>Document Intelligence Platform</p>
#         </div>
#     """, unsafe_allow_html=True)
    
#     # Check backend connectivity
#     if not check_backend_health():
#         st.error("⚠️ **Backend service is not accessible.** Please ensure the backend is running on port 8000.")
#         st.stop()
    
#     # Enhanced sidebar
#     with st.sidebar:
#         st.markdown("### 🧭 Navigation")
#         page = option_menu(
#             menu_title=None,
#             options=["💬 Chat", "📤 Upload", "📊 Analytics", "📁 Documents"],
#             icons=["chat-dots", "cloud-upload", "bar-chart", "files"],
#             default_index=0,
#             orientation="vertical",
#             styles={
#                 "container": {"padding": "0"},
#                 "icon": {"color": "#667eea", "font-size": "18px"},
#                 "nav-link": {
#                     "font-size": "16px",
#                     "text-align": "left",
#                     "margin": "5px",
#                     "padding": "10px 15px",
#                     "border-radius": "10px"
#                 },
#                 "nav-link-selected": {
#                     "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
#                     "color": "white"
#                 }
#             }
#         )
        
#         st.markdown("---")
        
#         # System status with enhanced styling
#         st.markdown("### 🔧 System Status")
#         if check_backend_health():
#             st.markdown(
#                 '<div class="metric-card">✅ Backend Online<br><small>All systems operational</small></div>',
#                 unsafe_allow_html=True
#             )
#         else:
#             st.error("❌ Backend Offline")
        
#         # Document count with styling
#         docs = get_documents()
#         st.markdown(
#             f'<div class="metric-card">📄 {len(docs)} Documents<br><small>Ready for analysis</small></div>',
#             unsafe_allow_html=True
#         )
        
#         # Quick tips
#         st.markdown("---")
#         st.markdown("### 💡 Quick Tips")
#         tips = [
#             "🚀 Upload CSV/Excel for best visualizations",
#             "🎨 Try different chart types in Analytics",
#             "💬 Ask specific questions for better results",
#             "📊 Use dashboard view for comprehensive analysis"
#         ]
#         for tip in tips:
#             st.markdown(f"<small>{tip}</small>", unsafe_allow_html=True)
    
#     # Route to appropriate page
#     page_name = page.split()[1]  # Extract page name without emoji
    
#     if page_name == "Chat":
#         render_chat_page()
#     elif page_name == "Upload":
#         render_upload_page()
#     elif page_name == "Analytics":
#         render_analytics_page()
#     elif page_name == "Documents":
#         render_documents_page()

# def render_chat_page():
#     """Enhanced chat interface with modern styling."""
#     st.markdown("## 💬 Chat with Your Documents")
    
#     # Enhanced chat input
#     col1, col2 = st.columns([4, 1])
#     with col1:
#         user_query = st.text_input(
#             "",
#             placeholder="💭 Ask anything about your documents... e.g., 'Show me sales trends by region'",
#             key="chat_input",
#             label_visibility="collapsed"
#         )
#     with col2:
#         include_viz = st.checkbox("🎨 Include Visualizations", value=True)
    
#     # Enhanced send button
#     if st.button("🚀 Send Message", type="primary", use_container_width=True) and user_query:
#         with st.spinner("🤖 AI is analyzing your query..."):
#             response_data = send_chat_message(user_query, include_viz)
            
#             if response_data:
#                 # Add to chat history
#                 st.session_state.chat_history.append({
#                     "role": "user",
#                     "content": user_query,
#                     "timestamp": datetime.now()
#                 })
#                 st.session_state.chat_history.append({
#                     "role": "assistant",
#                     "content": response_data["response"],
#                     "timestamp": datetime.now(),
#                     "sources": response_data.get("sources", []),
#                     "visualization": response_data.get("visualization")
#                 })
#         st.rerun()
    
#     # Display chat history with enhanced styling
#     if st.session_state.chat_history:
#         st.markdown("---")
#         for i, msg in enumerate(st.session_state.chat_history):
#             if msg["role"] == "user":
#                 st.markdown(f"""
#                     <div style='text-align: right; margin: 1rem 0;'>
#                         <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
#                                     color: white; padding: 1rem; border-radius: 15px 15px 5px 15px; 
#                                     display: inline-block; max-width: 70%; box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);'>
#                             {msg["content"]}
#                         </div>
#                     </div>
#                 """, unsafe_allow_html=True)
#             else:
#                 st.markdown(f"""
#                     <div style='text-align: left; margin: 1rem 0;'>
#                         <div style='background: #f8f9fa; color: #2c3e50; padding: 1rem; 
#                                     border-radius: 15px 15px 15px 5px; display: inline-block; 
#                                     max-width: 70%; border-left: 4px solid #4ECDC4;'>
#                             {msg["content"]}
#                         </div>
#                     </div>
#                 """, unsafe_allow_html=True)
                
#                 # Show sources
#                 if "sources" in msg and msg["sources"]:
#                     with st.expander("📚 Sources & References", expanded=False):
#                         for source in msg["sources"]:
#                             st.markdown(f"""
#                                 **📄 {source['filename']}** 
#                                 (Relevance: {source['relevance_score']:.1%})
                                
#                                 *{source['content'][:300]}...*
                                
#                                 ---
#                             """)
                
#                 # Show visualization
#                 if "visualization" in msg and msg["visualization"]:
#                     st.markdown("### 📊 Generated Visualization")
#                     render_plotly_visualization(msg["visualization"])
#     else:
#         # Welcome message with suggestions
#         st.markdown("""
#             <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
#                         border-radius: 1rem; margin: 2rem 0;'>
#                 <h3 style='color: #667eea;'>👋 Welcome to RAG Intelligence!</h3>
#                 <p>Upload some documents and start exploring your data with AI-powered insights.</p>
#             </div>
#         """, unsafe_allow_html=True)
        
#         if get_documents():
#             st.markdown("### 🌟 Suggested Questions")
#             suggestions = [
#                 "📈 What are the main trends in my data?",
#                 "🔍 Summarize the key findings from my documents",
#                 "📊 Show me a comparison between different categories",
#                 "🎯 Create a visualization of performance metrics"
#             ]
            
#             cols = st.columns(2)
#             for i, suggestion in enumerate(suggestions):
#                 with cols[i % 2]:
#                     if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
#                         # Process suggestion automatically
#                         with st.spinner("🤖 Processing suggestion..."):
#                             response_data = send_chat_message(suggestion, True)
#                             if response_data:
#                                 st.session_state.chat_history.append({
#                                     "role": "user",
#                                     "content": suggestion,
#                                     "timestamp": datetime.now()
#                                 })
#                                 st.session_state.chat_history.append({
#                                     "role": "assistant",
#                                     "content": response_data["response"],
#                                     "timestamp": datetime.now(),
#                                     "sources": response_data.get("sources", []),
#                                     "visualization": response_data.get("visualization")
#                                 })
#                         st.rerun()

# def render_upload_page():
#     """Enhanced file upload interface with modern design."""
#     st.markdown("## 📤 Upload Your Documents")
    
#     # Enhanced upload area
#     st.markdown("""
#         <div class="upload-section">
#             <h3 style="margin-top: 0; color: #667eea;">🎯 Drag & Drop Your Files</h3>
#             <p style="color: #6c757d; margin-bottom: 2rem;">Support for multiple formats: PDF, Word, Excel, CSV, PowerPoint, JSON</p>
#         </div>
#     """, unsafe_allow_html=True)
    
#     uploaded_files = st.file_uploader(
#         "Choose files to upload",
#         type=['pdf', 'docx', 'csv', 'xlsx', 'txt', 'json', 'pptx'],
#         accept_multiple_files=True,
#         help="💡 For best visualizations, upload CSV or Excel files with numeric data",
#         label_visibility="collapsed"
#     )
    
#     if uploaded_files:
#         st.markdown("### 📋 Selected Files")
        
#         # Display files in a nice format
#         for file in uploaded_files:
#             file_size = file.size / 1024  # Convert to KB
#             size_unit = "KB" if file_size < 1024 else f"{file_size/1024:.1f} MB"
#             size_display = f"{file_size:.1f} {size_unit}" if file_size < 1024 else size_unit
            
#             st.markdown(f"""
#                 <div style="background: #f8f9fa; padding: 1rem; margin: 0.5rem 0; 
#                            border-radius: 0.5rem; border-left: 4px solid #4ECDC4;">
#                     📄 <strong>{file.name}</strong> 
#                     <span style="color: #6c757d;">({size_display})</span>
#                 </div>
#             """, unsafe_allow_html=True)
        
#         # Enhanced upload button
#         if st.button("🚀 Process Files", type="primary", use_container_width=True):
#             with st.spinner("🔄 Processing your files... This may take a moment."):
#                 progress_bar = st.progress(0)
                
#                 # Simulate progress
#                 for i in range(100):
#                     time.sleep(0.01)
#                     progress_bar.progress(i + 1)
                
#                 result = upload_files(uploaded_files)
#                 progress_bar.empty()
                
#                 if result and result["success"]:
#                     st.success(f"🎉 {result['message']}")
#                     st.session_state.uploaded_files.extend(result["files"])
                    
#                     # Enhanced processing results
#                     st.markdown("### 📊 Processing Results")
                    
#                     for file_info in result["files"]:
#                         with st.expander(f"📄 {file_info['filename']}", expanded=True):
#                             col1, col2, col3 = st.columns(3)
                            
#                             with col1:
#                                 st.metric("File Type", file_info['file_type'].upper())
#                                 st.metric("Text Chunks", file_info['chunk_count'])
                            
#                             with col2:
#                                 st.metric("Processing Time", f"{file_info['processing_time']:.2f}s")
#                                 if 'metadata' in file_info:
#                                     file_size = file_info['metadata'].get('file_size', 0)
#                                     st.metric("File Size", f"{file_size:,} bytes")
                            
#                             with col3:
#                                 # Show file type specific info
#                                 metadata = file_info.get('metadata', {})
#                                 if file_info['file_type'] == '.csv':
#                                     st.metric("Data Rows", metadata.get('rows', 'N/A'))
#                                     st.metric("Columns", metadata.get('columns', 'N/A'))
#                                 elif file_info['file_type'] == '.pdf':
#                                     st.metric("Pages", metadata.get('pages', 'N/A'))
#                                 elif file_info['file_type'] == '.docx':
#                                     st.metric("Paragraphs", metadata.get('paragraphs', 'N/A'))
                    
#                     # Auto-refresh after upload
#                     time.sleep(2)
#                     st.rerun()
    
#     # Enhanced tips section
#     st.markdown("---")
#     st.markdown("### 💡 Optimization Tips")
    
#     tips_data = [
#         ("📊 CSV/Excel Files", "Perfect for creating interactive charts and analytics dashboards", "#4ECDC4"),
#         ("📄 PDF/Word Documents", "Excellent for text analysis, Q&A, and content summarization", "#45B7D1"),
#         ("📈 PowerPoint Files", "Great for extracting presentation content and slide analysis", "#96CEB4"),
#         ("📝 JSON/Text Files", "Ideal for structured data analysis and pattern recognition", "#FFEAA7")
#     ]
    
#     for title, description, color in tips_data:
#         st.markdown(f"""
#             <div style="background: linear-gradient(135deg, {color}20 0%, {color}10 100%); 
#                         padding: 1rem; margin: 0.5rem 0; border-radius: 0.75rem; 
#                         border-left: 4px solid {color};">
#                 <strong style="color: {color};">{title}</strong><br>
#                 <small style="color: #2c3e50;">{description}</small>
#             </div>
#         """, unsafe_allow_html=True)

# def render_analytics_page():
#     """Enhanced analytics dashboard with modern Plotly visualizations."""
#     st.markdown("## 📊 Advanced Analytics Dashboard")
    
#     docs = get_documents()
#     if not docs:
#         st.markdown("""
#             <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%); 
#                         border-radius: 1rem; border: 2px dashed #fc8181;'>
#                 <h3 style='color: #e53e3e; margin-bottom: 1rem;'>📤 No Documents Found</h3>
#                 <p style='color: #c53030;'>Please upload some documents first to generate analytics.</p>
#                 <p style='color: #c53030;'>💡 CSV and Excel files work best for data visualizations!</p>
#             </div>
#         """, unsafe_allow_html=True)
#         return

#     # Quick Visualization Section
#     st.markdown("### 🎨 Quick Visualization Generator")
    
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         viz_query = st.text_input(
#             "",
#             placeholder="🎯 e.g., 'Show a bar chart of sales by region' or 'Create a trend line for monthly revenue'",
#             key="viz_query",
#             label_visibility="collapsed"
#         )
    
#     with col2:
#         chart_style = st.selectbox(
#             "Style",
#             ["Modern", "Professional", "Vibrant", "Ocean", "Sunset"],
#             key="chart_style"
#         )

#     if st.button("🎨 Generate Visualization", type="primary", use_container_width=True):
#         if viz_query:
#             with st.spinner("🎨 Creating your visualization..."):
#                 viz_result = get_visualization(viz_query)
                
#                 if viz_result:
#                     if "error" in viz_result:
#                         st.error(f"❌ {viz_result['error']}")
#                     else:
#                         st.success("✨ Visualization generated successfully!")
#                         render_plotly_visualization(viz_result)
#                 else:
#                     st.error("❌ Failed to generate visualization. Try a different query.")
#         else:
#             st.warning("💭 Please enter a visualization request.")

#     # Comprehensive Analytics Section
#     st.markdown("---")
#     st.markdown("### 📈 Comprehensive Analytics Suite")
    
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         analytics_query = st.text_input(
#             "",
#             placeholder="🔍 e.g., 'Analyze sales performance trends' or 'Show customer satisfaction patterns'",
#             key="analytics_query",
#             label_visibility="collapsed"
#         )
    
#     with col2:
#         chart_types = st.multiselect(
#             "Chart Types",
#             ["bar", "line", "scatter", "histogram", "pie", "heatmap", "box"],
#             default=["bar", "line", "scatter"],
#             key="chart_types_select"
#         )

#     if st.button("🚀 Generate Complete Analytics", use_container_width=True):
#         if analytics_query:
#             with st.spinner("🤖 AI is analyzing your data... This may take a moment."):
#                 analytics_result = get_analytics(analytics_query, chart_types)
                
#                 if analytics_result and "error" not in analytics_result:
#                     # Data Summary Dashboard
#                     if "data_summary" in analytics_result and isinstance(analytics_result["data_summary"], dict):
#                         st.markdown("### 📊 Data Overview Dashboard")
#                         summary = analytics_result["data_summary"]
                        
#                         col1, col2, col3, col4 = st.columns(4)
                        
#                         metrics = [
#                             ("Total Rows", summary.get("total_rows", "N/A"), "📊"),
#                             ("Total Columns", summary.get("total_columns", "N/A"), "📋"),
#                             ("Numeric Columns", len(summary.get("numeric_columns", [])), "🔢"),
#                             ("Text Columns", len(summary.get("categorical_columns", [])), "📝")
#                         ]
                        
#                         colors = ["#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
                        
#                         for i, ((title, value, icon), color) in enumerate(zip(metrics, colors)):
#                             with [col1, col2, col3, col4][i]:
#                                 st.markdown(f"""
#                                     <div style="background: linear-gradient(135deg, {color}30 0%, {color}10 100%); 
#                                                 padding: 1.5rem; border-radius: 1rem; text-align: center; 
#                                                 border: 2px solid {color}30;">
#                                         <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
#                                         <div style="font-size: 2rem; font-weight: bold; color: {color}; margin-bottom: 0.25rem;">{value:,}</div>
#                                         <div style="color: #2c3e50; font-weight: 500;">{title}</div>
#                                     </div>
#                                 """, unsafe_allow_html=True)

#                     # Visualizations Section
#                     if "visualizations" in analytics_result and analytics_result["visualizations"]:
#                         st.markdown("### 📈 Interactive Visualizations")
                        
#                         # Display visualizations in a grid
#                         viz_count = len(analytics_result["visualizations"])
#                         cols_per_row = 2 if viz_count > 1 else 1
                        
#                         for i in range(0, viz_count, cols_per_row):
#                             cols = st.columns(cols_per_row)
                            
#                             for j in range(cols_per_row):
#                                 if i + j < viz_count:
#                                     with cols[j]:
#                                         viz = analytics_result["visualizations"][i + j]
#                                         st.markdown(f"#### {viz.get('description', f'Chart {i+j+1}')}")
#                                         render_plotly_visualization(viz)

#                     # Insights Section
#                     if "insights" in analytics_result and analytics_result["insights"]:
#                         display_insights(analytics_result["insights"])

#                 else:
#                     error_msg = analytics_result.get("error", "Unknown error") if analytics_result else "No response from backend"
#                     st.error(f"❌ Analytics generation failed: {error_msg}")
#         else:
#             st.warning("💭 Please enter an analytics query.")

#     # Quick Analytics Shortcuts
#     st.markdown("---")
#     st.markdown("### ⚡ Quick Analytics")
    
#     quick_options = [
#         ("📊 Data Overview", "Provide a comprehensive overview of all my data", "#4ECDC4"),
#         ("📈 Trend Analysis", "Show me trends and patterns in the data", "#45B7D1"),
#         ("🔗 Correlation Analysis", "Analyze relationships between variables", "#96CEB4"),
#         ("📉 Distribution Analysis", "Show the distribution of key variables", "#FFEAA7")
#     ]
    
#     cols = st.columns(2)
#     for i, (title, query, color) in enumerate(quick_options):
#         with cols[i % 2]:
#             if st.button(title, key=f"quick_{i}", use_container_width=True):
#                 with st.spinner(f"🤖 Generating {title.lower()}..."):
#                     analytics_result = get_analytics(query, ["bar", "line", "histogram", "scatter"])
                    
#                     if analytics_result and "error" not in analytics_result:
#                         st.success(f"✅ {title} completed!")
                        
#                         # Show quick results
#                         if "visualizations" in analytics_result:
#                             for viz in analytics_result["visualizations"][:2]:  # Show first 2 charts
#                                 render_plotly_visualization(viz)
                        
#                         if "insights" in analytics_result:
#                             with st.expander("💡 Key Insights", expanded=True):
#                                 for insight in analytics_result["insights"][:5]:
#                                     st.markdown(f"• {insight}")
#                     else:
#                         st.error(f"❌ Failed to generate {title.lower()}")

# def render_documents_page():
#     """Enhanced documents management with modern design."""
#     st.markdown("## 📁 Document Management Center")
    
#     docs = get_documents()
    
#     if not docs:
#         st.markdown("""
#             <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
#                         border-radius: 1rem; border: 2px dashed #0ea5e9;'>
#                 <h3 style='color: #0284c7; margin-bottom: 1rem;'>📁 No Documents Yet</h3>
#                 <p style='color: #0369a1;'>Upload some documents to get started with AI-powered analysis.</p>
#             </div>
#         """, unsafe_allow_html=True)
#         return
    
#     # Enhanced Statistics Dashboard
#     st.markdown("### 📊 Document Statistics")
#     total_docs = len(docs)
#     total_chunks = sum(doc.get("chunk_count", 0) for doc in docs)
#     total_size = sum(doc.get("file_size", 0) for doc in docs)
#     avg_processing_time = sum(doc.get("processing_time", 0) for doc in docs) / len(docs)
    
#     col1, col2, col3, col4 = st.columns(4)
    
#     metrics = [
#         ("Documents", total_docs, "📄", "#4ECDC4"),
#         ("Text Chunks", total_chunks, "📝", "#45B7D1"),
#         ("Total Size", f"{total_size/1024/1024:.1f} MB", "💾", "#96CEB4"),
#         ("Avg Process Time", f"{avg_processing_time:.1f}s", "⚡", "#FFEAA7")
#     ]
    
#     for i, (title, value, icon, color) in enumerate(metrics):
#         with [col1, col2, col3, col4][i]:
#             st.markdown(f"""
#                 <div style="background: linear-gradient(135deg, {color}20 0%, {color}10 100%); 
#                             padding: 1.5rem; border-radius: 1rem; text-align: center; 
#                             border-left: 4px solid {color};">
#                     <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
#                     <div style="font-size: 1.8rem; font-weight: bold; color: {color}; margin-bottom: 0.25rem;">{value}</div>
#                     <div style="color: #2c3e50; font-size: 0.9rem;">{title}</div>
#                 </div>
#             """, unsafe_allow_html=True)
    
#     # Documents List
#     st.markdown("### 📋 Document Library")
    
#     # Add search functionality
#     search_term = st.text_input("🔍 Search documents...", placeholder="Enter filename or type")
    
#     filtered_docs = docs
#     if search_term:
#         filtered_docs = [doc for doc in docs if search_term.lower() in doc.get('filename', '').lower()]
    
#     for doc in filtered_docs:
#         with st.expander(f"📄 {doc['filename']}", expanded=False):
#             col1, col2, col3 = st.columns([2, 2, 1])
            
#             with col1:
#                 st.markdown("**📋 Basic Info**")
#                 st.write(f"**File ID:** `{doc['file_id']}`")
#                 st.write(f"**Type:** {doc['file_type'].upper()}")
#                 st.write(f"**Chunks:** {doc['chunk_count']:,}")
            
#             with col2:
#                 st.markdown("**📊 Processing Stats**")
#                 st.write(f"**Size:** {doc['file_size']:,} bytes")
#                 st.write(f"**Processing Time:** {doc['processing_time']:.2f}s")
                
#                 # Processing efficiency
#                 efficiency = doc['chunk_count'] / max(doc['processing_time'], 0.1)
#                 st.write(f"**Efficiency:** {efficiency:.1f} chunks/sec")
            
#             with col3:
#                 st.markdown("**🔧 Actions**")
#                 if st.button("🗑️ Delete", key=f"delete_{doc['file_id']}", 
#                            type="secondary", use_container_width=True):
#                     if delete_document(doc['file_id']):
#                         st.success(f"✅ Deleted {doc['filename']}")
#                         time.sleep(1)
#                         st.rerun()
    
#     # Bulk Operations
#     st.markdown("---")
#     st.markdown("### 🔧 Bulk Operations")
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         if st.button("🔄 Refresh List", use_container_width=True):
#             st.rerun()
    
#     with col2:
#         if st.button("📊 Generate Report", use_container_width=True):
#             # Generate a simple report
#             report = f"""
#             ## 📊 Document Analysis Report
            
#             **Total Documents:** {len(docs)}
#             **Total Chunks:** {total_chunks:,}
#             **Average Chunks per Document:** {total_chunks/len(docs):.1f}
#             **Total Storage:** {total_size/1024/1024:.2f} MB
            
#             **File Types:**
#             """
            
#             file_types = {}
#             for doc in docs:
#                 file_type = doc.get('file_type', 'unknown')
#                 file_types[file_type] = file_types.get(file_type, 0) + 1
            
#             for file_type, count in file_types.items():
#                 report += f"\n- {file_type.upper()}: {count} files"
            
#             st.markdown(report)
    
#     with col3:
#         if st.button("🗑️ Clear All Documents", type="secondary", use_container_width=True):
#             if st.session_state.get("confirm_clear_all"):
#                 deleted_count = 0
#                 for doc in docs:
#                     if delete_document(doc['file_id']):
#                         deleted_count += 1
                
#                 st.success(f"✅ Deleted {deleted_count} documents")
#                 st.session_state.confirm_clear_all = False
#                 time.sleep(1)
#                 st.rerun()
#             else:
#                 st.session_state.confirm_clear_all = True
#                 st.warning("⚠️ Click again to confirm deletion of ALL documents")

# if __name__ == "__main__":
#     main()
import streamlit as st
import requests
import json
import os
import base64
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from streamlit_option_menu import option_menu
from streamlit_chat import message
import time

# Configuration - Fixed port to match original setup
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8002")

# Page configuration
st.set_page_config(
    page_title="RAG System - Document Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "current_documents" not in st.session_state:
    st.session_state.current_documents = []

def check_backend_health():
    """Check if backend is accessible"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_files(files):
    """Upload files to backend"""
    try:
        files_data = []
        for file in files:
            files_data.append(("files", (file.name, file.getvalue(), file.type)))
        
        response = requests.post(f"{BACKEND_URL}/upload", files=files_data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error uploading files: {str(e)}")
        return None

def send_chat_message(query, include_viz=True):
    """Send chat message to backend"""
    try:
        chat_data = {
            "query": query,
            "chat_history": [
                {"role": msg["role"], "content": msg["content"]} 
                for msg in st.session_state.chat_history[-10:]  # Last 10 messages
            ],
            "include_visualization": include_viz,
            "max_results": 5
        }
        
        response = requests.post(f"{BACKEND_URL}/chat", json=chat_data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error sending message: {str(e)}")
        return None

def get_analytics(query, chart_types=None):
    """Get analytics from backend"""
    try:
        analytics_data = {
            "query": query,
            "chart_types": chart_types or ["bar", "line", "scatter", "histogram"]
        }
        
        response = requests.post(f"{BACKEND_URL}/analytics", json=analytics_data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error getting analytics: {str(e)}")
        return None

def get_documents():
    """Get list of uploaded documents"""
    try:
        response = requests.get(f"{BACKEND_URL}/documents")
        response.raise_for_status()
        return response.json()["documents"]
    except Exception as e:
        st.error(f"Error getting documents: {str(e)}")
        return []

def get_visualization(query):
    """Get visualization from backend"""
    try:
        viz_data = {"query": query}
        response = requests.post(f"{BACKEND_URL}/visualization", json=viz_data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")
        return None

def delete_document(file_id):
    """Delete a document"""
    try:
        response = requests.delete(f"{BACKEND_URL}/documents/{file_id}")
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error deleting document: {str(e)}")
        return False

def render_visualization(viz_data):
    """Render visualization from backend response"""
    if not viz_data:
        return
    
    try:
        if viz_data.get("type") == "image":
            # Handle base64 image
            image_data = viz_data.get("data", "")
            if image_data:
                if image_data.startswith("data:image"):
                    # Remove data URL prefix
                    image_data = image_data.split(",")[1]
                
                # Decode and display
                image_bytes = base64.b64decode(image_data)
                st.image(image_bytes, caption=viz_data.get("description", "Visualization"), use_column_width=True)
            else:
                st.warning("No image data found in visualization")
                
        elif viz_data.get("type") in ["chart", "table"]:
            # Handle Plotly charts
            try:
                fig = go.Figure(viz_data["data"])
                st.plotly_chart(fig, use_container_width=True)
                st.caption(viz_data.get("description", "Chart"))
            except Exception as e:
                st.error(f"Error rendering Plotly chart: {str(e)}")
        else:
            st.warning(f"Unknown visualization type: {viz_data.get('type')}")
            
    except Exception as e:
        st.error(f"Error rendering visualization: {str(e)}")

def display_base64_image(base64_str, caption="Chart"):
    """Display base64 encoded image"""
    try:
        if base64_str.startswith("data:image"):
            base64_str = base64_str.split(",")[1]
        
        image_bytes = base64.b64decode(base64_str)
        st.image(image_bytes, caption=caption, use_column_width=True)
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")

def main():
    # Header
    st.title("🤖 RAG System - Document Intelligence")
    st.markdown("*Upload documents, ask questions, get insights with interactive visualizations*")
    
    # Check backend connectivity
    if not check_backend_health():
        st.error("⚠️ Backend service is not accessible. Please ensure the backend is running.")
        return
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        page = option_menu(
            menu_title=None,
            options=["Chat", "Upload", "Analytics", "Documents"],
            icons=["chat-dots", "cloud-upload", "bar-chart", "files"],
            default_index=0,
            orientation="vertical"
        )
        
        st.markdown("---")
        st.header("System Status")
        if check_backend_health():
            st.success("✅ Backend Online")
        else:
            st.error("❌ Backend Offline")
        
        # Document count
        docs = get_documents()
        st.info(f"📄 {len(docs)} Documents Loaded")
    
    # Main content based on selected page
    if page == "Chat":
        render_chat_page()
    elif page == "Upload":
        render_upload_page()
    elif page == "Analytics":
        render_analytics_page()
    elif page == "Documents":
        render_documents_page()

def render_chat_page():
    """Render the chat interface"""
    st.header("💬 Chat with Your Documents")
    
    # Chat input
    col1, col2 = st.columns([4, 1])
    with col1:
        user_query = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., What are the main trends in this sales data?",
            key="chat_input"
        )
    with col2:
        include_viz = st.checkbox("Include Visualizations", value=True)
    
    # Send button
    if st.button("Send", type="primary", use_container_width=True) and user_query:
        with st.spinner("Processing your question..."):
            response_data = send_chat_message(user_query, include_viz)
            
            if response_data:
                # Add to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_query,
                    "timestamp": datetime.now()
                })
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_data["response"],
                    "timestamp": datetime.now(),
                    "sources": response_data.get("sources", []),
                    "visualization": response_data.get("visualization")
                })
        
        # Clear input and rerun
        st.rerun()
    
    # Display chat history
    st.markdown("---")
    
    if st.session_state.chat_history:
        chat_container = st.container()
        with chat_container:
            for i, msg in enumerate(st.session_state.chat_history):
                if msg["role"] == "user":
                    message(msg["content"], is_user=True, key=f"user_{i}")
                else:
                    message(msg["content"], is_user=False, key=f"assistant_{i}")
                    
                    # Show sources if available
                    if "sources" in msg and msg["sources"]:
                        with st.expander("📚 Sources", expanded=False):
                            for source in msg["sources"]:
                                st.write(f"**{source['filename']}** (Relevance: {source['relevance_score']:.3f})")
                                st.write(f"*{source['content'][:200]}...*")
                                st.markdown("---")
                    
                    # Show visualization if available
                    if "visualization" in msg and msg["visualization"]:
                        st.markdown("**📊 Visualization:**")
                        render_visualization(msg["visualization"])
    else:
        st.info("👋 Welcome! Upload some documents and start asking questions.")
    
    # Quick suggestions
    if not st.session_state.chat_history and get_documents():
        st.markdown("### 💡 Quick Suggestions")
        suggestions = [
            "Summarize the key findings from my documents",
            "What are the main trends in the data?",
            "Show me a comparison between different categories",
            "Create a visualization of the performance metrics"
        ]
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    with st.spinner("Processing..."):
                        response_data = send_chat_message(suggestion, True)
                        if response_data:
                            st.session_state.chat_history.append({
                                "role": "user",
                                "content": suggestion,
                                "timestamp": datetime.now()
                            })
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": response_data["response"],
                                "timestamp": datetime.now(),
                                "sources": response_data.get("sources", []),
                                "visualization": response_data.get("visualization")
                            })
                    st.rerun()

def render_upload_page():
    """Render the file upload interface"""
    st.header("📤 Upload Documents")
    
    # Upload area
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['pdf', 'docx', 'csv', 'xlsx', 'txt', 'json', 'pptx'],
        accept_multiple_files=True,
        help="Supported formats: PDF, Word, Excel, CSV, Text, JSON, PowerPoint"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_files:
        st.write(f"**Selected {len(uploaded_files)} files:**")
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size} bytes)")
        
        if st.button("Upload Files", type="primary", use_container_width=True):
            with st.spinner("Processing files..."):
                result = upload_files(uploaded_files)
                
                if result and result["success"]:
                    st.success(f"✅ {result['message']}")
                    st.session_state.uploaded_files.extend(result["files"])
                    
                    # Show processing results
                    st.markdown("### Processing Results")
                    for file_info in result["files"]:
                        with st.expander(f"📄 {file_info['filename']}", expanded=False):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Type:** {file_info['file_type']}")
                                st.write(f"**Chunks:** {file_info['chunk_count']}")
                            with col2:
                                st.write(f"**Processing Time:** {file_info['processing_time']:.2f}s")
                                if 'metadata' in file_info:
                                    st.write(f"**File Size:** {file_info['metadata'].get('file_size', 'N/A')} bytes")
                    
                    time.sleep(2)
                    st.rerun()
    
    # Upload tips
    st.markdown("---")
    st.markdown("### 💡 Upload Tips")
    tips = [
        "📊 **CSV/Excel files**: Perfect for data analysis and visualizations",
        "📄 **PDF/Word documents**: Great for text analysis and Q&A",
        "📈 **PowerPoint**: Useful for extracting presentation content",
        "📝 **Text/JSON files**: Suitable for structured data analysis"
    ]
    for tip in tips:
        st.markdown(tip)

def render_analytics_page():
    """Render the analytics dashboard with integrated Visualization Service"""
    st.header("📊 Advanced Analytics")
    
    docs = get_documents()
    if not docs:
        st.warning("📤 Please upload some documents first to generate analytics.")
        return

    # Two main sections
    st.markdown("### 🎯 Quick Visualization")
    
    # Quick visualization section
    viz_query = st.text_input(
        "Enter a query for visualization:",
        placeholder="e.g., Show a bar chart of sales by region, Create a line chart showing trends over time",
        key="viz_query"
    )

    if st.button("Generate Visualization", type="primary", use_container_width=True):
        if viz_query:
            with st.spinner("Generating visualization..."):
                viz_result = get_visualization(viz_query)
                
                if viz_result:
                    if "error" in viz_result:
                        st.error(f"Visualization error: {viz_result['error']}")
                    elif "chart_base64" in viz_result:
                        st.success("✅ Visualization generated successfully!")
                        display_base64_image(viz_result["chart_base64"], viz_result.get("description", "Custom Visualization"))
                    else:
                        st.warning("No visualization data returned from backend.")
                else:
                    st.error("Failed to generate visualization.")
        else:
            st.warning("Please enter a query for visualization.")

    st.markdown("---")
    st.markdown("### 📈 Comprehensive Analytics")

    # Comprehensive analytics section
    col1, col2 = st.columns([3, 1])
    with col1:
        analytics_query = st.text_input(
            "What kind of analysis would you like?",
            placeholder="e.g., Generate comprehensive analytics for sales performance",
            key="analytics_query"
        )
    with col2:
        chart_types = st.multiselect(
            "Chart Types",
            ["bar", "line", "scatter", "histogram", "pie", "heatmap", "box"],
            default=["bar", "line", "scatter"],
            key="chart_types_select"
        )

    # Generate Analytics button
    if st.button("Generate Analytics", key="analytics_button", use_container_width=True):
        if analytics_query:
            with st.spinner("Generating comprehensive analytics..."):
                analytics_result = get_analytics(analytics_query, chart_types)
                
                if analytics_result and "error" not in analytics_result:
                    # Display data summary
                    if "data_summary" in analytics_result:
                        st.markdown("### 📈 Data Summary")
                        summary = analytics_result["data_summary"]
                        
                        if isinstance(summary, dict) and "error" not in summary:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Rows", summary.get("total_rows", "N/A"))
                            with col2:
                                st.metric("Total Columns", summary.get("total_columns", "N/A"))
                            with col3:
                                st.metric("Numeric Columns", len(summary.get("numeric_columns", [])))
                            with col4:
                                st.metric("Categorical Columns", len(summary.get("categorical_columns", [])))

                    # Display visualizations
                    if "visualizations" in analytics_result and analytics_result["visualizations"]:
                        st.markdown("### 📊 Generated Visualizations")
                        for i, viz in enumerate(analytics_result["visualizations"]):
                            st.markdown(f"#### {viz.get('description', f'Chart {i+1}')}")
                            
                            if viz.get("type") == "image" or "data:image" in str(viz.get("data", "")):
                                # Handle base64 image
                                image_data = viz.get("data", "")
                                if image_data:
                                    display_base64_image(image_data, viz.get("description", f"Chart {i+1}"))
                            else:
                                # Handle other visualization types
                                render_visualization(viz)
                            
                            st.markdown("---")

                    # # Display insights
                    if "insights" in analytics_result and analytics_result["insights"]:
                        st.markdown("### 💡 Key Insights")
                        for insight in analytics_result["insights"]:
                            st.write(f"• {insight}")

                    # Additional analytics data
                    if "distribution_plot" in analytics_result and analytics_result["distribution_plot"]:
                        st.markdown("### 📈 Distribution Analysis")
                        display_base64_image(analytics_result["distribution_plot"], "Distribution Plot")

                    if "correlation_heatmap" in analytics_result and analytics_result["correlation_heatmap"]:
                        st.markdown("### 🔥 Correlation Heatmap")
                        display_base64_image(analytics_result["correlation_heatmap"], "Correlation Heatmap")

                else:
                    error_msg = analytics_result.get("error", "Unknown error") if analytics_result else "No response from backend"
                    st.error(f"Failed to generate analytics: {error_msg}")
        else:
            st.warning("Please enter an analytics query.")

    # Quick Analytics Options
    st.markdown("---")
    st.markdown("### 🚀 Quick Analytics Options")
    quick_options = [
        ("Data Overview", "Provide a comprehensive overview of all my data"),
        ("Trend Analysis", "Show me trends and patterns in the data"),
        ("Correlation Analysis", "Analyze correlations between different variables"),
        ("Distribution Analysis", "Show the distribution of key variables")
    ]
    
    cols = st.columns(2)
    for i, (title, query) in enumerate(quick_options):
        with cols[i % 2]:
            if st.button(title, key=f"quick_{i}", use_container_width=True):
                with st.spinner(f"Generating {title.lower()}..."):
                    analytics_result = get_analytics(query, ["bar", "line", "histogram", "scatter"])
                    
                    if analytics_result and "error" not in analytics_result:
                        st.success(f"✅ {title} generated successfully!")
                        # Display results inline
                        if "visualizations" in analytics_result:
                            for viz in analytics_result["visualizations"]:
                                if viz.get("data"):
                                    display_base64_image(viz["data"], viz.get("description", title))
                    else:
                        error_msg = analytics_result.get("error", "Unknown error") if analytics_result else "No response"
                        st.error(f"Failed to generate {title.lower()}: {error_msg}")

def render_documents_page():
    """Render the documents management page"""
    st.header("📁 Document Management")
    
    docs = get_documents()
    
    if not docs:
        st.info("📤 No documents uploaded yet. Go to the Upload page to add some documents.")
        return
    
    # Document statistics
    st.markdown("### 📊 Document Statistics")
    total_docs = len(docs)
    total_chunks = sum(doc.get("chunk_count", 0) for doc in docs)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", total_docs)
    with col2:
        st.metric("Total Chunks", total_chunks)
    with col3:
        avg_chunks = total_chunks / total_docs if total_docs > 0 else 0
        st.metric("Avg Chunks/Doc", f"{avg_chunks:.1f}")
    
    # Documents table
    st.markdown("### 📄 Uploaded Documents")
    
    for doc in docs:
        with st.expander(f"📄 {doc['filename']}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**File ID:** {doc['file_id']}")
                st.write(f"**Type:** {doc['file_type']}")
                st.write(f"**Chunks:** {doc['chunk_count']}")
                st.write(f"**Size:** {doc['file_size']} bytes")
                st.write(f"**Processing Time:** {doc['processing_time']:.2f}s")
            
            with col2:
                if st.button("🗑️ Delete", key=f"delete_{doc['file_id']}", type="secondary"):
                    if delete_document(doc['file_id']):
                        st.success(f"✅ Deleted {doc['filename']}")
                        time.sleep(1)
                        st.rerun()
    
    # Bulk operations
    st.markdown("---")
    st.markdown("### 🔧 Bulk Operations")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Document List", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear All Documents", type="secondary", use_container_width=True):
            if st.session_state.get("confirm_clear_all"):
                # Perform deletion
                deleted_count = 0
                for doc in docs:
                    if delete_document(doc['file_id']):
                        deleted_count += 1
                
                st.success(f"✅ Deleted {deleted_count} documents")
                st.session_state.confirm_clear_all = False
                time.sleep(1)
                st.rerun()
            else:
                st.session_state.confirm_clear_all = True
                st.warning("⚠️ Click again to confirm deletion of all documents")

if __name__ == "__main__":
    main()
