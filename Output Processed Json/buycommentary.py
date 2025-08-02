#!/usr/bin/env python3
"""
DualCast - Buy Phase Commentary System
Real-time eSports commentary generator for Valorant buy phases
"""

import json
import time
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import logging
from elevenlabs.client import ElevenLabs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dualcast_commentary.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Add ElevenLabs API Key ---
ELEVENLABS_API_KEY = "sk_a4a4e9f07cf89c9b8c0ac00f784c778200ef67715c83fb8b" # API KEY HERE 
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

class EventType(Enum):
    REQUESTING_WEAPON = "requesting_weapon"
    TEAMMATE_REQUESTING = "teammate_requesting_weapon"
    WEAPON_OWNED = "weapon_owned"
    SHIELD_OWNED = "shield_owned"
    ABILITY_OWNED = "ability_owned"
    SIDE_ARM_OWNED = "side_arm_owned"
    WEAPON_HOVER = "weapon_hover"

class CasterRole(Enum):
    HYPE = "Hype"
    ANALYST = "Analyst"

@dataclass
class Event:
    timestamp: float
    event_type: str
    player: str
    weapon: Optional[str] = None
    team: Optional[str] = None
    teammate: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Event':
        return cls(
            timestamp=data.get('timestamp', time.time()),
            event_type=data.get('type', ''),
            player=data.get('player', ''),
            weapon=data.get('weapon'),
            team=data.get('team'),
            teammate=data.get('teammate')
        )

@dataclass
class CommentaryLine:
    timestamp: float
    caster: str
    text: str
    event: Event

class CommentaryTemplates:
    """Manages all commentary templates for different event types and caster roles"""
    
    TEMPLATES = {
        EventType.REQUESTING_WEAPON: {
            CasterRole.HYPE: [
                "${player} wants a ${weapon}! Can the team hook him up?",
                "Big request from ${player} — they're going all in on that ${weapon}!",
                "Yo, ${player} asking for a ${weapon} drop! Someone give this man a gun!",
                "${player} is going full send with a ${weapon} request — this could be huge!",
                "That's a clutch call from ${player}. He wants a ${weapon} — let's see if someone answers.",
                "${player} calling for backup! They need a ${weapon} ASAP!",
                "Request incoming! ${player} wants that ${weapon} — team chemistry time!",
                "Look at ${player} going for broke! ${weapon} request on the table!",
                "${player} trusting the squad — big ${weapon} ask right here!",
                "Communication is key! ${player} wants a ${weapon} — who's got their back?",
                "${player} making moves! That ${weapon} request could change everything!",
                "Team play alert! ${player} needs that ${weapon} — time to deliver!",
                "Bold call from ${player}! ${weapon} request — this is what teamwork looks like!",
                "${player} going for the dream! ${weapon} drop request — let's see the magic!",
                "Strategic call! ${player} asking for a ${weapon} — the team better respond!"
            ],
            CasterRole.ANALYST: [
                "${player} requests a ${weapon}. Seems like they're low on credits.",
                "Smart call by ${player}, asking for a ${weapon} instead of forcing.",
                "${player} trusting the team economy — requesting a ${weapon}.",
                "This request could decide the round. ${player} needs a ${weapon}.",
                "${player} needs a ${weapon}. The team's economy better be ready.",
                "Economic management in play — ${player} requesting a ${weapon}.",
                "${player} making the right call here, asking for a ${weapon} drop.",
                "Team coordination test — ${player} wants a ${weapon}.",
                "${player} conserving credits by requesting a ${weapon}.",
                "Strategic thinking from ${player} — ${weapon} request shows discipline.",
                "${player} reading the economy well — asking for that ${weapon}.",
                "Good fundamentals from ${player} — requesting rather than forcing the buy.",
                "${player} prioritizing team resources with this ${weapon} request.",
                "Smart economic play — ${player} asks for a ${weapon} instead of overbuying.",
                "${player} showing patience with the ${weapon} request — textbook teamwork."
            ]
        },
        EventType.TEAMMATE_REQUESTING: {
            CasterRole.HYPE: [
                "${teammate} pings for a ${weapon} — who's stepping up?",
                "Team needs to rally — ${teammate} wants a buy!",
                "Request on the board! ${teammate} is asking for a ${weapon}!",
                "${teammate} calling for help! ${weapon} needed!",
                "Team chemistry check! ${teammate} wants that ${weapon}!",
                "${teammate} putting trust in the squad — ${weapon} request!",
                "Big ask from ${teammate}! Who's got the ${weapon}?",
                "${teammate} needs support! ${weapon} request incoming!",
                "Teamwork time! ${teammate} asking for a ${weapon}!",
                "${teammate} making the call — ${weapon} drop needed!",
                "Communication flowing! ${teammate} wants a ${weapon}!",
                "${teammate} trusting the process — ${weapon} request up!",
                "Squad support! ${teammate} calling for that ${weapon}!"
            ],
            CasterRole.ANALYST: [
                "${teammate} asking for a ${weapon}. That might affect team balance.",
                "${teammate} sending a buy request — watch for a potential drop.",
                "Could be a sign of an eco round — ${teammate} requesting help.",
                "${teammate} coordinating with the team — ${weapon} request.",
                "Team economy in motion — ${teammate} asking for support.",
                "${teammate} managing resources — requesting a ${weapon}.",
                "Strategic communication from ${teammate} — ${weapon} needed.",
                "${teammate} showing economic discipline with this request.",
                "Team dynamic at work — ${teammate} asking for a ${weapon}.",
                "${teammate} reading the situation — ${weapon} request makes sense."
            ]
        },
        EventType.WEAPON_OWNED: {
            CasterRole.HYPE: [
                "${player} locks in the ${weapon}! Let's go!",
                "Vandal buy from ${player} — they're ready to pop off!",
                "${player} ain't messing around — ${weapon} in hand!",
                "Full commitment! ${player} grabs the ${weapon}!",
                "${player} locked and loaded with that ${weapon}!",
                "Business time! ${player} secures the ${weapon}!",
                "${player} going for glory — ${weapon} purchased!",
                "No hesitation! ${player} takes the ${weapon}!",
                "${player} ready for action — ${weapon} equipped!",
                "Investment made! ${player} commits to the ${weapon}!",
                "${player} showing confidence — ${weapon} in the arsenal!",
                "Game time! ${player} locks in that ${weapon}!",
                "${player} means business — ${weapon} secured!",
                "All in! ${player} grabs the ${weapon}!",
                "${player} ready to rumble with that ${weapon}!"
            ],
            CasterRole.ANALYST: [
                "${player} picks up a ${weapon}. Looks like a full buy.",
                "${team} investing early — ${player} goes with ${weapon}.",
                "${player} confirms a ${weapon} purchase. Strong start.",
                "Standard buy from ${player} — ${weapon} secured.",
                "${player} following the game plan with a ${weapon}.",
                "Economic stability shown — ${player} buys the ${weapon}.",
                "${player} making the expected play — ${weapon} purchase.",
                "Team economy allows ${player} to grab that ${weapon}.",
                "${player} prioritizing firepower with the ${weapon}.",
                "Solid investment from ${player} — ${weapon} in hand.",
                "${player} optimizing their loadout with the ${weapon}.",
                "Tactical purchase — ${player} secures the ${weapon}.",
                "${player} sticking to fundamentals — ${weapon} buy.",
                "Meta play from ${player} — ${weapon} selected.",
                "${player} making the percentage play with the ${weapon}."
            ]
        },
        EventType.SHIELD_OWNED: {
            CasterRole.HYPE: [
                "${player} grabs armor — full commit!",
                "${player} just armored up. That's confidence!",
                "Shields up! ${player} is ready for war!",
                "${player} going tank mode — armor secured!",
                "Defense first! ${player} grabs the protection!",
                "${player} investing in survival — armor on!",
                "Smart move! ${player} prioritizes armor!",
                "${player} playing it safe — shields equipped!",
                "Armor check! ${player} is covered!",
                "${player} preparing for battle — armor ready!",
                "Protection mode! ${player} gets armored up!",
                "${player} thinking ahead — armor purchased!"
            ],
            CasterRole.ANALYST: [
                "${player} buys armor. Indicates they're playing serious.",
                "${team} going for survivability — ${player} grabs shields.",
                "${player} prioritizing sustainability with armor.",
                "Standard defensive buy — ${player} gets armor.",
                "${player} investing in staying power with shields.",
                "${player} making the right call — armor first.",
                "Defensive-minded purchase from ${player} — armor secured.",
                "${player} showing discipline with the armor buy.",
                "Practical approach from ${player} — shields equipped.",
                "${player} optimizing survival chances with armor."
            ]
        },
        EventType.ABILITY_OWNED: {
            CasterRole.HYPE: [
                "${player} loading utility — they're cooking something!",
                "${player} fully stocked with abilities. Watch out!",
                "Utility time! ${player} is prepared!",
                "${player} grabbing the tools — abilities ready!",
                "Setup potential! ${player} loads utility!",
                "${player} going for the big brain plays — abilities secured!",
                "Tactical genius! ${player} stocks up on utility!",
                "${player} ready to create chaos — abilities equipped!",
                "Game-changing potential! ${player} grabs utility!",
                "${player} preparing the magic — abilities loaded!",
                "Strategic depth! ${player} secures utility!",
                "${player} bringing the toolkit — abilities ready!"
            ],
            CasterRole.ANALYST: [
                "${player} grabs their kit — probably setting up for control.",
                "Utility purchase from ${player}. Might see a setup play.",
                "${player} investing in map control with abilities.",
                "Strategic buy from ${player} — utility secured.",
                "${player} prioritizing team utility with this purchase.",
                "${player} preparing for tactical plays — abilities ready.",
                "Setup potential from ${player} with utility purchase.",
                "${player} showing tactical awareness — abilities equipped.",
                "Team play incoming — ${player} grabs utility.",
                "${player} investing in round control with abilities."
            ]
        },
        EventType.SIDE_ARM_OWNED: {
            CasterRole.HYPE: [
                "${player} just bought a ${weapon}? Eco alert!",
                "${player} on a ${weapon} — risky but bold!",
                "Pistol power! ${player} keeps it light!",
                "${player} going minimalist — ${weapon} only!",
                "Eco round energy! ${player} grabs a ${weapon}!",
                "${player} showing restraint — ${weapon} purchase!",
                "Budget buy! ${player} sticks with the ${weapon}!",
                "${player} playing it careful — ${weapon} secured!",
                "Light buy from ${player} — ${weapon} in hand!",
                "${player} keeping it simple — ${weapon} only!",
                "Economic discipline! ${player} takes the ${weapon}!"
            ],
            CasterRole.ANALYST: [
                "${player} keeps it light — just a ${weapon}.",
                "Looks like a save — ${player} picks a ${weapon}.",
                "${player} conserving credits with a ${weapon} buy.",
                "Economic management from ${player} — ${weapon} only.",
                "${player} playing the long game — ${weapon} purchase.",
                "Disciplined buy from ${player} — just the ${weapon}.",
                "${player} saving for next round — ${weapon} equipped.",
                "Smart economy from ${player} — ${weapon} secured.",
                "${player} avoiding overcommitment — ${weapon} buy.",
                "Tactical restraint from ${player} — ${weapon} only."
            ]
        },
        EventType.WEAPON_HOVER: {
            CasterRole.HYPE: [
                "${player} hovering a ${weapon}. Will they buy?",
                "Indecision from ${player} — maybe switching things up?",
                "${player} considering options — ${weapon} in sight!",
                "Decision time! ${player} eyeing that ${weapon}!",
                "${player} weighing their choices — ${weapon} hovered!",
                "Thinking it through! ${player} looks at the ${weapon}!",
                "${player} on the fence — ${weapon} consideration!",
                "Big decision! ${player} hovering the ${weapon}!",
                "${player} taking their time — ${weapon} in view!",
                "Choice to make! ${player} eyes the ${weapon}!",
                "${player} deliberating — ${weapon} under consideration!"
            ],
            CasterRole.ANALYST: [
                "${player} considering a ${weapon}. Not committed yet.",
                "Possible buy from ${player}. Let's see if they lock it.",
                "${player} evaluating their options — ${weapon} hovered.",
                "Decision-making process from ${player} — ${weapon} considered.",
                "${player} weighing the economic impact — ${weapon} hover.",
                "Strategic thinking from ${player} — ${weapon} under review.",
                "${player} taking time to decide — ${weapon} in consideration.",
                "Tactical evaluation from ${player} — ${weapon} hovered.",
                "${player} assessing the situation — ${weapon} potential buy.",
                "Economic calculation from ${player} — ${weapon} consideration."
            ]
        }
    }
    
    @classmethod
    def get_template(cls, event_type: EventType, caster_role: CasterRole) -> str:
        """Get a random template for the given event type and caster role"""
        templates = cls.TEMPLATES.get(event_type, {}).get(caster_role, [])
        if not templates:
            return f"{caster_role.value}: Event occurred for {event_type.value}"
        return random.choice(templates)
    
    @classmethod
    def format_template(cls, template: str, event: Event) -> str:
        """Replace template variables with actual values"""
        replacements = {
            'player': event.player,
            'weapon': event.weapon or 'weapon',
            'team': event.team or 'team',
            'teammate': event.teammate or 'teammate'
        }
        
        result = template
        for key, value in replacements.items():
            result = result.replace(f'${{{key}}}', value)
        
        return result

class BuyPhaseCommentator:
    """Main commentary system that processes events and generates commentary"""
    
    # Event priority mapping (lower number = higher priority)
    EVENT_PRIORITY = {
        'requesting_weapon': 1,
        'teammate_requesting_weapon': 2,
        'weapon_owned': 3,
        'shield_owned': 4,
        'ability_owned': 5,
        'side_arm_owned': 6,
        'weapon_hover': 7
    }
    
    def __init__(self, json_file_path: str = "buy_phase_events.json"):
        self.json_file_path = json_file_path
        self.processed_events: List[Event] = []
        self.commentary_lines: List[CommentaryLine] = []
        self.last_caster_role = CasterRole.ANALYST  # Start with analyst so first is hype
        self.last_modification_time = 0
        self.phase_start_time = None
        self.phase_duration = 10  # seconds
        self.commentary_interval = 2  # seconds between commentary lines
        self.last_commentary_time = 0
        self.player_mention_count = {}  # Track mentions per player
        self.running = False
        
        # Create empty JSON file if it doesn't exist
        self._initialize_json_file()
    
    def _initialize_json_file(self):
        """Create an empty JSON file if it doesn't exist"""
        if not os.path.exists(self.json_file_path):
            with open(self.json_file_path, 'w') as f:
                json.dump([], f)
            logger.info(f"Created empty JSON file: {self.json_file_path}")
    
    def _read_events_from_file(self) -> List[Event]:
        """Read and parse events from the JSON file"""
        try:
            if not os.path.exists(self.json_file_path):
                return []
            
            # Check if file was modified
            current_mod_time = os.path.getmtime(self.json_file_path)
            if current_mod_time <= self.last_modification_time:
                return []
            
            self.last_modification_time = current_mod_time
            
            with open(self.json_file_path, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                logger.warning("JSON file should contain a list of events")
                return []
            
            events = []
            for item in data:
                try:
                    event = Event.from_dict(item)
                    events.append(event)
                except Exception as e:
                    logger.warning(f"Failed to parse event: {item}, error: {e}")
            
            return events
            
        except Exception as e:
            logger.error(f"Error reading events from file: {e}")
            return []
    
    def _should_process_event(self, event: Event) -> bool:
        """Determine if an event should be processed"""
        # Check for duplicates
        for processed in self.processed_events:
            if (processed.player == event.player and 
                processed.event_type == event.event_type and
                abs(processed.timestamp - event.timestamp) < 1.0):
                return False
        
        # Don't process hover events if we have a purchase for the same player/weapon
        if event.event_type == 'weapon_hover':
            for processed in self.processed_events:
                if (processed.player == event.player and 
                    processed.weapon == event.weapon and
                    processed.event_type in ['weapon_owned', 'requesting_weapon']):
                    return False
        
        return True
    
    def _get_next_caster_role(self) -> CasterRole:
        """Alternate between caster roles"""
        if self.last_caster_role == CasterRole.HYPE:
            self.last_caster_role = CasterRole.ANALYST
        else:
            self.last_caster_role = CasterRole.HYPE
        return self.last_caster_role
    
    def _generate_commentary_line(self, event: Event) -> CommentaryLine:
        """Generate a commentary line for an event"""
        try:
            event_type = EventType(event.event_type)
        except ValueError:
            logger.warning(f"Unknown event type: {event.event_type}")
            event_type = EventType.WEAPON_OWNED  # Default fallback
        
        caster_role = self._get_next_caster_role()
        template = CommentaryTemplates.get_template(event_type, caster_role)
        text = CommentaryTemplates.format_template(template, event)
        
        return CommentaryLine(
            timestamp=time.time(),
            caster=caster_role.value,
            text=text,
            event=event
        )
    
    def _prioritize_events(self, events: List[Event]) -> List[Event]:
        """Sort events by priority and apply filtering logic"""
        # Filter out events that shouldn't be processed
        valid_events = [e for e in events if self._should_process_event(e)]
        
        # Sort by priority (lower number = higher priority)
        sorted_events = sorted(valid_events, key=lambda e: self.EVENT_PRIORITY.get(e.event_type, 999))
        
        # Limit to top 3-4 events and balance player mentions
        selected_events = []
        player_counts = {}
        
        for event in sorted_events:
            if len(selected_events) >= 4:
                break
            
            # Avoid focusing too much on one player
            current_count = player_counts.get(event.player, 0)
            if current_count >= 2:  # Max 2 mentions per player per phase
                continue
            
            selected_events.append(event)
            player_counts[event.player] = current_count + 1
        
        return selected_events
    
    def _should_generate_commentary(self) -> bool:
        """Check if enough time has passed to generate new commentary"""
        current_time = time.time()
        return (current_time - self.last_commentary_time) >= self.commentary_interval
    
    def process_buy_phase(self):
        """Main processing loop for a buy phase"""
        if self.phase_start_time is None:
            self.phase_start_time = time.time()
            logger.info("Buy phase started - monitoring for events...")
        
        current_time = time.time()
        phase_elapsed = current_time - self.phase_start_time
        
        # Check if buy phase is over
        if phase_elapsed > self.phase_duration:
            logger.info("Buy phase completed")
            return False
        
        # Read new events
        new_events = self._read_events_from_file()
        
        if new_events:
            logger.info(f"Found {len(new_events)} new events")
            
            # Prioritize and select events
            selected_events = self._prioritize_events(new_events)
            
            # Generate commentary if enough time has passed
            if selected_events and self._should_generate_commentary():
                event = selected_events[0]  # Take highest priority event
                commentary = self._generate_commentary_line(event)
                
                self.commentary_lines.append(commentary)
                self.processed_events.append(event)
                self.last_commentary_time = current_time
                
                # Output commentary
                self._output_commentary(commentary)
        
        return True

    def _speak_commentary(self, text: str, caster_role: CasterRole):
        """Converts text to speech and plays it using ElevenLabs."""
        try:
            # Select a voice based on the caster role
            voice_id = "Adam" if caster_role == CasterRole.ANALYST else "Josh"
            
            # Generate audio using the client's TTS stream capability
            audio_stream = client.generate(
                text=text,
                voice=voice_id,
                model="eleven_multilingual_v2"
            )
            
            # The 'stream' function from the library plays the audio stream
            from elevenlabs import stream
            stream(audio_stream)
            
        except Exception as e:
            logger.error(f"Failed to generate or play audio: {e}")

    def _output_commentary(self, commentary: CommentaryLine):
        """Output commentary line to console and log"""
        output = f"[{commentary.caster}] {commentary.text}"
        print(f"\n🎙️  {output}")
        logger.info(f"Commentary: {output}")

        # --- Add this line to speak the commentary ---
        self._speak_commentary(commentary.text, CasterRole(commentary.caster))
    
    def start_monitoring(self):
        """Start the main monitoring loop"""
        self.running = True
        logger.info("Starting DualCast Buy Phase Commentary System...")
        print("🎮 DualCast Commentary System Started!")
        print("📁 Monitoring:", self.json_file_path)
        print("⏱️  Buy Phase Duration:", self.phase_duration, "seconds")
        print("🎯 Commentary Interval:", self.commentary_interval, "seconds")
        print("=" * 60)
        
        try:
            while self.running:
                if not self.process_buy_phase():
                    # Buy phase ended, reset for next phase
                    self.reset_phase()
                    time.sleep(1)  # Wait before next phase
                else:
                    time.sleep(0.5)  # Check every 500ms
                    
        except KeyboardInterrupt:
            logger.info("System stopped by user")
            print("\n👋 Commentary system stopped.")
        except Exception as e:
            logger.error(f"System error: {e}")
            print(f"\n❌ System error: {e}")
    
    def reset_phase(self):
        """Reset system for next buy phase"""
        self.phase_start_time = None
        self.processed_events.clear()
        self.commentary_lines.clear()
        self.player_mention_count.clear()
        self.last_commentary_time = 0
        logger.info("Phase reset - ready for next buy phase")
        print("\n🔄 Ready for next buy phase...")
        print("=" * 60)
    
    def stop(self):
        """Stop the monitoring system"""
        self.running = False
    
    def get_commentary_summary(self) -> Dict:
        """Get summary of current phase commentary"""
        return {
            "phase_duration": time.time() - self.phase_start_time if self.phase_start_time else 0,
            "events_processed": len(self.processed_events),
            "commentary_lines": len(self.commentary_lines),
            "commentary": [
                {
                    "timestamp": line.timestamp,
                    "caster": line.caster,
                    "text": line.text,
                    "event_type": line.event.event_type,
                    "player": line.event.player
                }
                for line in self.commentary_lines
            ]
        }

def create_sample_events():
    """Create sample events for testing"""
    sample_events = [
        {
            "timestamp": time.time(),
            "type": "requesting_weapon",
            "player": "TenZ",
            "weapon": "Vandal",
            "team": "Sentinels"
        },
        {
            "timestamp": time.time() + 1,
            "type": "weapon_owned",
            "player": "Zyppan",
            "weapon": "Phantom",
            "team": "NAVI"
        },
        {
            "timestamp": time.time() + 2,
            "type": "shield_owned",
            "player": "TenZ",
            "team": "Sentinels"
        },
        {
            "timestamp": time.time() + 3,
            "type": "teammate_requesting_weapon",
            "player": "Derke",
            "teammate": "Derke",
            "weapon": "Operator",
            "team": "FNATIC"
        },
        {
            "timestamp": time.time() + 4,
            "type": "weapon_hover",
            "player": "aspas",
            "weapon": "Vandal",
            "team": "LEV"
        }
    ]
    
    with open("buy_phase_events.json", "w") as f:
        json.dump(sample_events, f, indent=2)
    
    print("📝 Sample events created in buy_phase_events.json")

def main():
    """Main function to run the commentary system"""
    print("🎮 DualCast - Buy Phase Commentary System")
    print("=" * 50)
    
    # Ask user if they want to create sample events
    response = input("Create sample events for testing? (y/n): ").lower().strip()
    if response == 'y':
        create_sample_events()
        print("✅ Sample events created!")
    
    # Initialize and start the commentary system
    commentator = BuyPhaseCommentator()
    
    try:
        commentator.start_monitoring()
    except Exception as e:
        logger.error(f"Failed to start commentary system: {e}")
        print(f"❌ Failed to start system: {e}")

if __name__ == "__main__":
    main()
