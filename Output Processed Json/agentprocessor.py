import json
import time
import random
from pathlib import Path
from fuzzywuzzy import fuzz
from typing import Dict, List, Optional, Tuple
import threading
import queue

class ValorantCommentarySystem:
    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.last_modified = 0
        self.current_agent = None
        self.is_locked_in = False
        self.commentary_queue = queue.Queue()
        
        # Valorant agent names for fuzzy matching
        self.valorant_agents = [
            "ASTRA", "BREACH", "BRIMSTONE", "CYPHER", "JETT", "KILLJOY",
            "OMEN", "PHOENIX", "RAZE", "REYNA", "SAGE", "SKYE", "SOVA",
            "VIPER", "YORU", "KAY/O", "CHAMBER", "NEON", "FADE", "HARBOR",
            "GEKKO", "DEADLOCK", "ISO", "CLOVE", "VYSE"
        ]
        
        # Pre-defined commentaries for character selection (hovering/considering)
        self.selection_commentaries = [
            "They're eyeing [AGENT] — could this be the pick?",
            "[AGENT] is under consideration! Will they pull the trigger?",
            "Hovering over [AGENT] — the tension is building!",
            "They're thinking about [AGENT] — what a strategic choice that could be!",
            "[AGENT] is catching their attention — this could change everything!",
            "Considering [AGENT] — the team comp is starting to take shape!",
            "They're weighing their options with [AGENT] — smart thinking!",
            "[AGENT] in the spotlight — will they commit to this powerhouse?",
            "Looking at [AGENT] — the crowd is holding its breath!",
            "[AGENT] is tempting them — what a potential game-changer!"
        ]
        
        # Pre-defined commentaries for lock-in (your format)
        self.lockin_commentaries = [
            "[AGENT] is locked in! Things are about to heat up!",
            "Brace yourselves — [AGENT] just joined the fight!",
            "Oh my goodness, they're going with [AGENT]! This is going to be explosive!",
            "[AGENT] steps into the arena — the crowd is on their feet!",
            "[AGENT] locked and loaded! The battlefield just got more dangerous!",
            "It's official — [AGENT] is ready to dominate!",
            "[AGENT] has entered the game! What an incredible choice!",
            "They've committed to [AGENT] — this is going to be epic!",
            "[AGENT] is in the house! The energy is electric!",
            "Lock confirmed! [AGENT] is about to show what they're made of!"
        ]
        
        # Character-specific descriptions (add more as needed)
        self.agent_descriptions = {
            "ASTRA": "The cosmic controller Astra harnesses the energies of the cosmos to reshape battlefields. Her stellar abilities provide unparalleled map control.",
            "BREACH": "The bionic Swede Breach fires powerful, targeted kinetic blasts to aggressively clear a path through enemy ground.",
            "BRIMSTONE": "Joining from the USA, Brimstone's orbital arsenal ensures his squad always has the advantage. His ability to deliver utility precisely and from a distance makes him an unmatched commander on the battlefield.",
            "CYPHER": "The Moroccan information broker Cypher is a one-man surveillance network who keeps tabs on the enemy's every move.",
            "JETT": "Representing her home country of South Korea, the agile duelist Jett's playstyle is all about mobility and quick eliminations.",
            "KILLJOY": "The tech genius Killjoy secures the battlefield with ease. Her arsenal of inventions can hold down any site.",
            "OMEN": "A phantom of a memory, Omen hunts in the shadows. He renders enemies paranoid and strikes with precision.",
            "PHOENIX": "Hailing from the UK, Phoenix's star power shines through his fighting style, igniting the battlefield with flash and flare.",
            "RAZE": "Raze explodes out of Brazil with her big personality and big guns. She excels at flushing entrenched enemies out of hiding spots.",
            "REYNA": "Forged in the heart of Mexico, Reyna dominates single combat, popping off with each kill she scores.",
            "SAGE": "The stronghold of China, Sage provides unparalleled support for her team with her healing and resurrection abilities.",
            "SKYE": "Hailing from Australia, Skye and her band of beasts trail-blaze the way through hostile territory.",
            "SOVA": "Born from the eternal winter of Russia's tundra, Sova tracks down, finds, and eliminates enemies with ruthless efficiency.",
            "VIPER": "The American chemist Viper deploys an array of poisonous chemical devices to control the battlefield.",
            "YORU": "Japanese native Yoru rips holes straight through reality to infiltrate enemy lines unseen."
        }
    
    def normalize_agent_name(self, detected_name: str) -> Optional[str]:
        """Use fuzzy matching to find the closest Valorant agent name"""
        if not detected_name:
            return None
            
        detected_name = detected_name.upper().strip()
        
        # Direct match first
        if detected_name in self.valorant_agents:
            return detected_name
        
        # Fuzzy matching
        best_match = None
        best_ratio = 0
        
        for agent in self.valorant_agents:
            ratio = fuzz.ratio(detected_name, agent)
            if ratio > best_ratio and ratio >= 70:  # 70% similarity threshold
                best_ratio = ratio
                best_match = agent
        
        return best_match
    
    def parse_ocr_json(self, json_data: dict) -> Tuple[Optional[str], bool]:
        """Parse OCR JSON and extract agent name and lock-in status"""
        agent_name = None
        has_lockin_button = False
        
        try:
            results = json_data.get('results', [])
            
            for result in results:
                detections = result.get('detections', [])
                
                current_agent = None
                current_lockin = False
                
                for detection in detections:
                    class_name = detection.get('class_name', '')
                    ocr_result = detection.get('ocr_result', {})
                    text = ocr_result.get('text', '').strip()
                    
                    if class_name == 'Agent Names' and text:
                        normalized_agent = self.normalize_agent_name(text)
                        if normalized_agent:
                            current_agent = normalized_agent
                    
                    elif class_name == 'Pre LockIn' and text:
                        # If we detect "LOCK IN" or "LOCKIN" button, agent is not locked yet
                        if any(keyword in text.upper() for keyword in ['LOCK', 'LOCKIN']):
                            current_lockin = True
                
                # If we found both agent and lock-in button in the same image
                if current_agent and current_lockin:
                    agent_name = current_agent
                    has_lockin_button = True
                    break
        
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            return None, False
        
        return agent_name, has_lockin_button
    
    def generate_commentary(self, agent_name: str, is_selection: bool) -> str:
        """Generate appropriate commentary based on agent and state"""
        if is_selection:
            # Agent is being considered (lock-in button visible)
            base_commentary = random.choice(self.selection_commentaries)
            return base_commentary.replace("[AGENT]", agent_name)
        else:
            # Agent is locked in - use your commentary format
            lockin_commentary = random.choice(self.lockin_commentaries)
            commentary = lockin_commentary.replace("[AGENT]", agent_name)
            
            # Optionally add character description after the main commentary
            description = self.agent_descriptions.get(agent_name)
            if description:
                return f"{commentary}\n💡 {description}"
            else:
                return commentary
    
    def monitor_json_file(self):
        """Monitor the JSON file for changes and process updates"""
        print(f"Monitoring {self.json_file_path} for changes...")
        
        while True:
            try:
                file_path = Path(self.json_file_path)
                
                if not file_path.exists():
                    print(f"File {self.json_file_path} not found. Waiting...")
                    time.sleep(1)
                    continue
                
                current_modified = file_path.stat().st_mtime
                
                if current_modified > self.last_modified:
                    self.last_modified = current_modified
                    
                    with open(self.json_file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    agent_name, has_lockin_button = self.parse_ocr_json(json_data)
                    
                    if agent_name:
                        # Check if state changed
                        state_changed = False
                        
                        if agent_name != self.current_agent:
                            state_changed = True
                        elif has_lockin_button != (not self.is_locked_in):
                            state_changed = True
                        
                        if state_changed:
                            self.current_agent = agent_name
                            self.is_locked_in = not has_lockin_button
                            
                            commentary = self.generate_commentary(agent_name, has_lockin_button)
                            self.commentary_queue.put(commentary)
                            print(f"\n🎮 {commentary}")
                
                time.sleep(0.5)  # Check every 500ms
                
            except json.JSONDecodeError as e:
                print(f"Invalid JSON format: {e}")
                time.sleep(1)
            except Exception as e:
                print(f"Error monitoring file: {e}")
                time.sleep(1)
    
    def get_latest_commentary(self) -> Optional[str]:
        """Get the latest commentary from the queue"""
        try:
            return self.commentary_queue.get_nowait()
        except queue.Empty:
            return None
    
    def start_monitoring(self):
        """Start monitoring in a separate thread"""
        monitor_thread = threading.Thread(target=self.monitor_json_file, daemon=True)
        monitor_thread.start()
        return monitor_thread

# Example usage
if __name__ == "__main__":
    # Initialize the commentary system
    # Replace with your actual JSON file path
    json_file_path = "agent.json"  # Change this to your JSON file path
    
    commentary_system = ValorantCommentarySystem(json_file_path)
    
    # Start monitoring
    monitor_thread = commentary_system.start_monitoring()
    
    print("Valorant Commentary System Started!")
    print("Monitoring OCR data for agent selections and lock-ins...")
    print("Press Ctrl+C to stop")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping commentary system...")

# Additional utility functions for integration

def add_custom_commentary(commentary_system: ValorantCommentarySystem, new_commentaries: List[str], is_lockin: bool = True):
    """Add custom commentaries to the system
    
    Args:
        commentary_system: The commentary system instance
        new_commentaries: List of commentary strings with [AGENT] placeholder
        is_lockin: True for lock-in commentaries, False for selection commentaries
    """
    if is_lockin:
        commentary_system.lockin_commentaries.extend(new_commentaries)
    else:
        commentary_system.selection_commentaries.extend(new_commentaries)

def add_agent_description(commentary_system: ValorantCommentarySystem, agent_name: str, description: str):
    """Add or update agent description"""
    commentary_system.agent_descriptions[agent_name.upper()] = description

def get_current_state(commentary_system: ValorantCommentarySystem) -> Dict:
    """Get current state of the commentary system"""
    return {
        "current_agent": commentary_system.current_agent,
        "is_locked_in": commentary_system.is_locked_in,
        "last_modified": commentary_system.last_modified
    }