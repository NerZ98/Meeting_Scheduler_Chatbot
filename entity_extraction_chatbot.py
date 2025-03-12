import re
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import spacy
from spacy.matcher import Matcher

def setup_logger():
    """
    Configure comprehensive logging with both file and console handlers
    
    Returns:
        logging.Logger: Configured logger
        str: Path to the log file
    """
    # Create logs directory if it doesn't exist
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    
    # Generate unique log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(logs_dir, f'entity_extraction_{timestamp}.log')
    
    # Create logger
    logger = logging.getLogger('AdvancedEntityExtractor')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console Handler - for displaying important info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File Handler - for detailed logging
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger, log_filename

class AdvancedEntityExtractor:
    def __init__(self, logger=None):
        """
        Initialize the advanced entity extractor with multiple extraction strategies
        
        Args:
            logger (logging.Logger, optional): Logger instance. If None, creates a default logger.
        """
        # Setup logging
        if logger is None:
            self.logger, self.log_file = setup_logger()
        else:
            self.logger = logger
            self.log_file = None
        
        self.logger.info("Initializing Advanced Entity Extractor")
        
        try:
            # Load SpaCy model with transformer-based NER
            self.nlp = spacy.load('en_core_web_trf')
            self.logger.info("SpaCy transformer model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load SpaCy model: {e}")
            raise
        
        # Create a matcher for custom patterns
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_custom_patterns()
    
    def _setup_custom_patterns(self):
        """
        Setup custom pattern matching for specific entities
        """
        try:
            # Date patterns
            date_patterns = [
                [{'LOWER': 'tomorrow'}],
                [{'LOWER': 'today'}],
                [{'LOWER': 'next'}, {'LOWER': {'IN': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']}}]
            ]
            
            # Time patterns
            time_patterns = [
                [{'SHAPE': 'dd'}, {'LOWER': {'IN': ['am', 'pm']}}],
                [{'SHAPE': 'dd'}, {'LOWER': ':'},  {'SHAPE': 'dd'}, {'LOWER': {'IN': ['am', 'pm']}}]
            ]
            
            # Duration patterns
            duration_patterns = [
                [{'SHAPE': 'dd'}, {'LOWER': {'IN': ['mins', 'min', 'minutes', 'hours', 'hour']}}]
            ]
            
            self.logger.debug("Custom entity patterns setup completed")
        except Exception as e:
            self.logger.error(f"Error setting up custom patterns: {e}")
    
    def parse_date(self, date_str: str) -> str:
        """
        Convert relative dates to actual dates
        
        Args:
            date_str (str): Input date string
        
        Returns:
            str: Formatted date string
        """
        try:
            today = datetime.now()
            
            date_mapping = {
                'today': today,
                'tomorrow': today + timedelta(days=1),
                'yesterday': today - timedelta(days=1)
            }
            
            # Handle next day of week
            next_day_match = re.match(r'next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', date_str.lower())
            if next_day_match:
                target_day = next_day_match.group(1)
                days = {
                    'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 
                    'friday': 4, 'saturday': 5, 'sunday': 6
                }
                days_ahead = days[target_day] - today.weekday() + 7
                parsed_date = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
                self.logger.debug(f"Parsed next {target_day} to {parsed_date}")
                return parsed_date
            
            # Handle today, tomorrow, yesterday
            if date_str.lower() in date_mapping:
                parsed_date = date_mapping[date_str.lower()].strftime("%Y-%m-%d")
                self.logger.debug(f"Parsed {date_str} to {parsed_date}")
                return parsed_date
            
            return date_str
        except Exception as e:
            self.logger.error(f"Error parsing date '{date_str}': {e}")
            return date_str

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities using multiple strategies
        
        Args:
            text (str): Input text to extract entities from
        
        Returns:
            Dict[str, List[str]]: Extracted entities
        """
        self.logger.info(f"Extracting entities from text: '{text}'")
        
        # Initialize results dictionary
        entities = {
            "DATE": [],
            "TIME": [],
            "DURATION": [],
            "ATTENDEE": []
        }
        
        try:
            # Process text with SpaCy
            doc = self.nlp(text)
            
            # Date extraction
            date_patterns = [
                r'\b(?:today|tomorrow|yesterday)\b',
                r'\bnext\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b'
            ]
            date_regex = re.compile('|'.join(date_patterns), re.IGNORECASE)
            date_matches = date_regex.findall(text)
            entities['DATE'] = [self.parse_date(date) for date in date_matches]
            self.logger.debug(f"Extracted dates: {entities['DATE']}")
            
            # Time extraction
            time_patterns = [
                r'\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b',
                r'\b\d{1,2}\s*(?:am|pm)\b'
            ]
            time_regex = re.compile('|'.join(time_patterns), re.IGNORECASE)
            entities['TIME'] = time_regex.findall(text)
            self.logger.debug(f"Extracted times: {entities['TIME']}")
            
            # Duration extraction with multiple patterns
            duration_patterns = [
                # Patterns with units after number
                r'\b(\d+)\s*(?:minute|min|mins)\b',
                r'\b(\d+)\s*(?:hour|hr|hours)\b',
                
                # Patterns with 'for' before duration
                r'\bfor\s+(\d+)\s*(?:minute|min|mins)\b',
                r'\bfor\s+(\d+)\s*(?:hour|hr|hours)\b',
                
                # Less common variations
                r'\b(\d+)(?:m|min)\b',
                r'\b(\d+)(?:h|hr)\b'
            ]
            
            # Combine and find all matches
            full_duration_pattern = re.compile('|'.join(duration_patterns), re.IGNORECASE)
            duration_matches = full_duration_pattern.findall(text)
            
            # Process and format duration matches
            processed_durations = []
            for match in duration_matches:
                # Ensure we get the number (handle tuple results from regex)
                if isinstance(match, tuple):
                    # Take the first non-empty value
                    number = next((m for m in match if m), None)
                else:
                    number = match
                
                # Determine unit based on the original text
                if re.search(r'(?:minute|min|mins|m)(?!\w)', text, re.IGNORECASE):
                    processed_durations.append(f"{number} mins")
                elif re.search(r'(?:hour|hr|hours|h)(?!\w)', text, re.IGNORECASE):
                    processed_durations.append(f"{number} hours")
            
            entities['DURATION'] = processed_durations
            self.logger.debug(f"Extracted durations: {entities['DURATION']}")
            
            # Attendee extraction
            # First, try SpaCy NER for person names
            attendees = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
            
            # Fallback to custom name extraction
            if not attendees:
                name_pattern = r'\b[A-Z][a-z]+ (?:[A-Z][a-z]+)?\b'
                attendees = re.findall(name_pattern, text)
            
            # Remove duplicates while preserving order
            entities['ATTENDEE'] = list(dict.fromkeys(attendees))
            self.logger.debug(f"Extracted attendees: {entities['ATTENDEE']}")
            
            return entities
        
        except Exception as e:
            self.logger.error(f"Error during entity extraction: {e}")
            raise

def interactive_extractor():
    """
    Interactive entity extraction tool with logging
    """
    # Setup logging
    logger, log_file = setup_logger()
    
    try:
        # Create extractor with the logger
        extractor = AdvancedEntityExtractor(logger)
        
        logger.info("Starting Interactive Entity Extraction Tool")
        print("Advanced Entity Extraction Tool")
        print("Enter text to extract entities. Type 'exit' to quit.")
        print(f"Logging to: {log_file}")
        
        while True:
            # Get user input
            text = input("\nEnter text: ").strip()
            
            # Log the input
            logger.info(f"User input: {text}")
            
            # Check for exit condition
            if text.lower() == 'exit':
                logger.info("Exiting interactive tool")
                print("Exiting...")
                break
            
            # Extract and display entities
            try:
                results = extractor.extract_entities(text)
                
                print("\nExtracted Entities:")
                for category, values in results.items():
                    if values:
                        print(f"  {category}: {values}")
                    else:
                        print(f"  {category}: None")
                
                # Log successful extraction
                logger.info("Entity extraction completed successfully")
            
            except Exception as e:
                logger.error(f"Extraction error: {e}")
                print(f"An error occurred: {e}")
    
    except KeyboardInterrupt:
        logger.info("Interactive tool interrupted by user")
        print("\nOperation cancelled by user.")
    
    except Exception as e:
        logger.critical(f"Unexpected error in interactive tool: {e}")
        print(f"A critical error occurred: {e}")

def main():
    """
    Main function to run the extractor
    """
    print("""
Before running, install required libraries:
pip install spacy
python -m spacy download en_core_web_trf
""")
    
    interactive_extractor()

if __name__ == "__main__":
    main()