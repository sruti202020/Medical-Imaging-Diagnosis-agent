

import streamlit as st
from datetime import datetime
import json
import os
import uuid
import time
import openai

# Chat system storage
def get_chat_store():
    """Get the chat storage"""
    if os.path.exists("chat_store.json"):
        with open("chat_store.json", "r") as f:
            return json.load(f)
    return {"rooms": {}}

def save_chat_store(store):
    """Save the chat storage"""
    with open("chat_store.json", "w") as f:
        json.dump(store, f)

def create_chat_room(case_id, creator_name, case_description):
    """Create a new chat room for a case"""
    store = get_chat_store()
    
    # Generate a unique room ID if one doesn't exist
    if case_id not in store["rooms"]:
        room_data = {
            "id": case_id,
            "created_at": datetime.now().isoformat(),
            "creator": creator_name,
            "description": case_description,
            "participants": [creator_name, "Dr. AI Assistant", "Dr. Johnson", "Dr. Chen", "Dr. Patel"],
            "messages": []
        }
        # Add an initial welcome message from AI
        welcome_message = {
            "id": str(uuid.uuid4()),
            "user": "Dr. AI Assistant",
            "content": f"Welcome to the case discussion for '{case_description}'. I've analyzed the image and I'm here to assist with the diagnosis. Feel free to ask me specific questions about the findings.",
            "type": "text",
            "timestamp": datetime.now().isoformat()
        }
        room_data["messages"].append(welcome_message)
        
        store["rooms"][case_id] = room_data
        save_chat_store(store)
    
    return case_id

def join_chat_room(case_id, user_name):
    """Join an existing chat room"""
    store = get_chat_store()
    
    if case_id in store["rooms"]:
        if user_name not in store["rooms"][case_id]["participants"]:
            store["rooms"][case_id]["participants"].append(user_name)
            save_chat_store(store)
        return True
    
    return False

def add_message(case_id, user_name, message, message_type="text"):
    """Add a message to a chat room"""
    store = get_chat_store()
    
    if case_id in store["rooms"]:
        message_data = {
            "id": str(uuid.uuid4()),
            "user": user_name,
            "content": message,
            "type": message_type,
            "timestamp": datetime.now().isoformat()
        }
        store["rooms"][case_id]["messages"].append(message_data)
        save_chat_store(store)
        return message_data
    
    return None




def get_available_rooms():
    """Get a list of all available chat rooms"""
    store = get_chat_store()
    rooms = []
    
    for room_id, room_data in store["rooms"].items():
        rooms.append({
            "id": room_id,
            "description": room_data["description"],
            "creator": room_data["creator"],
            "created_at": room_data["created_at"],
            "participants": len(room_data["participants"])
        })
    
    # Sort by creation date (newest first)
    rooms.sort(key=lambda x: x["created_at"], reverse=True)
    return rooms

def get_openai_response(user_question, case_description, findings=None, api_key=None):
    """Get a response from OpenAI based on the medical context and user question"""
    if not api_key:
        return "Please configure your OpenAI API key in the sidebar to get AI responses."
    
    # Set up OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    # Create the findings text if available
    findings_text = ""
    if findings and len(findings) > 0:
        findings_text = "The key findings in the image are:\n"
        for i, finding in enumerate(findings, 1):
            findings_text += f"{i}. {finding}\n"
    
    # Create system prompt with medical context
    system_prompt = f"""You are Dr. AI Assistant, a medical specialist analyzing a medical image. 
    The image is from a case described as: "{case_description}". 
    {findings_text}
    
    Please provide an expert, accurate, and helpful response to the doctor's question.
    Base your response on the findings and your medical expertise.
    Respond as if you are speaking directly to the doctor in a collaborative setting.
    Keep your response concise but informative, focusing on the relevant medical details.
    """
    
    try:
        # Make the API call to OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # You can use "gpt-4" for more advanced responses
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            max_tokens=300,
            temperature=0.2,  # Lower temperature for more consistent medical responses
        )
        
        # Extract and return the response text
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return f"I apologize, but I encountered an error while analyzing your question. Please try again or rephrase your question. Error details: {str(e)}"

def render_chat_interface():
    """Render the chat interface in the Streamlit app"""
    st.subheader("👨‍⚕️👩‍⚕️ Multi-Doctor Collaboration")
    
    # User information
    if "user_name" not in st.session_state:
        st.session_state.user_name = "Dr. Anonymous"
    
    user_name = st.text_input("Your Name", value=st.session_state.user_name)
    if user_name != st.session_state.user_name:
        st.session_state.user_name = user_name
    
    # Case selection or creation
    tab1, tab2 = st.tabs(["Join Existing Case", "Create New Case"])
    
    with tab1:
        # Show available chat rooms
        rooms = get_available_rooms()
        if rooms:
            room_options = {f"{room['id']} - {room['description']} (by {room['creator']})": room["id"] for room in rooms}
            selected_room = st.selectbox("Select Case", options=list(room_options.keys()))
            
            if st.button("Join Discussion"):
                selected_case_id = room_options[selected_room]
                if join_chat_room(selected_case_id, user_name):
                    st.session_state.current_case_id = selected_case_id
                    st.rerun()
        else:
            st.info("No active case discussions. Create a new one!")
    
    with tab2:
        # Create a new chat room
        case_description = st.text_input("Case Description")
        
        # FIX: Check if file_type exists in session_state before accessing it
        can_create_discussion = "file_data" in st.session_state and "file_type" in st.session_state and st.session_state.file_type is not None
        
        if can_create_discussion:
            # Now we can safely use file_type since we know it exists and is not None
            case_id = f"{st.session_state.file_type.upper()}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            if st.button("Create Discussion"):
                if case_description:
                    created_case_id = create_chat_room(case_id, user_name, case_description)
                    st.session_state.current_case_id = created_case_id
                    st.rerun()
                else:
                    st.error("Please provide a case description")
        else:
            # Provide more informative message
            if "file_data" not in st.session_state:
                st.info("Upload an image first to create a new case discussion")
            elif "file_type" not in st.session_state or st.session_state.file_type is None:
                st.info("Please complete the image upload and processing before creating a discussion")
            else:
                st.info("Upload an image first to create a new case discussion")
    
    # Active chat display
    if "current_case_id" in st.session_state:
        case_id = st.session_state.current_case_id
        store = get_chat_store()
        
        if case_id in store["rooms"]:
            room_data = store["rooms"][case_id]
            
            # Display chat header
            st.subheader(f"Case Discussion: {room_data['description']}")
            st.caption(f"Created by {room_data['creator']} • {len(room_data['participants'])} participants")
            
            # Response options
            response_col1, response_col2 = st.columns(2)
            with response_col1:
                get_ai_response = st.checkbox("Get AI Assistant Response", value=True)
            with response_col2:
                if get_ai_response:
                    doctor_response = False
                else:
                    doctor_response = st.checkbox("Get Doctor Response", value=True)
                    if doctor_response:
                        doctor_name = st.selectbox("Select Doctor", 
                                                   ["Dr. Johnson (Cardiologist)", 
                                                    "Dr. Chen (Pulmonologist)", 
                                                    "Dr. Patel (Radiologist)"])
                        # Extract just the name part
                        doctor_name = doctor_name.split(" (")[0]
            
            # Display messages
            messages = get_messages(case_id)
            
            chat_container = st.container()
            with chat_container:
                for msg in messages:
                    with st.chat_message(name=msg["user"], avatar="👨‍⚕️" if msg["user"] != user_name else "🧑‍⚕️"):
                        if msg["type"] == "text":
                            st.write(msg["content"])
                        elif msg["type"] == "annotation":
                            st.write("📝 **Image Annotation:**")
                            st.write(msg["content"])
            
            # Message input
            message = st.chat_input("Type your message here")
            if message:
                # Add user message
                add_message(case_id, user_name, message)
                
                # Get response from OpenAI or another doctor based on settings
                if get_ai_response:
                    # Add a small delay to simulate AI thinking
                    with st.spinner("AI Assistant is analyzing..."):
                        time.sleep(1)
                    
                    # Get findings if available
                    findings = st.session_state.get("findings", None)
                    
                    # Get API key from session state
                    api_key = st.session_state.get("OPENAI_API_KEY", None)
                    
                    # Generate and add AI response using OpenAI
                    ai_response = get_openai_response(message, room_data["description"], findings, api_key)
                    add_message(case_id, "Dr. AI Assistant", ai_response)
                
                elif doctor_response:
                    # Use simulated doctor responses (these could also be OpenAI generated with different prompts)
                    with st.spinner(f"{doctor_name} is typing..."):
                        time.sleep(1)
                    
                    # Simple doctor response generation
                    doctor_responses = {
                        "Dr. Johnson": "From a cardiac perspective, I'd want to rule out any cardiac involvement. The mild cardiomegaly noted in the image warrants further cardiac workup, possibly an echocardiogram.",
                        "Dr. Chen": "These infiltrates have a distribution pattern typical of atypical pneumonia. I'd recommend a sputum culture and respiratory pathogen panel to identify the causative agent.",
                        "Dr. Patel": "The radiographic findings show bilateral infiltrates with ground-glass opacities. This pattern is most consistent with an inflammatory process, likely infectious in etiology."
                    }
                    
                    doc_response = doctor_responses.get(doctor_name, "I concur with the assessment. Let's monitor the patient's response to treatment.")
                    add_message(case_id, doctor_name, doc_response)
                
                st.rerun()
            
            # Annotation option
            with st.expander("Add Image Annotation"):
                annotation = st.text_area("Describe what you see in the image")
                if st.button("Submit Annotation"):
                    add_message(case_id, user_name, annotation, message_type="annotation")
                    st.rerun()
        else:
            # Handle case where room no longer exists
            st.error("This case discussion no longer exists")
            if st.button("Return to Room Selection"):
                del st.session_state.current_case_id
                st.rerun()

# Manual creation function for usage without file upload
def create_manual_chat_room(creator_name, case_description):
    """Create a new chat room without requiring file upload"""
    # Generate a generic case ID
    case_id = f"CASE-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return create_chat_room(case_id, creator_name, case_description)