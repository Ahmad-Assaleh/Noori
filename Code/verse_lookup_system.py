"""
SIMPLIFIED QURAN LOOKUP SYSTEM
Step-by-step guide with explanations
"""

import xml.etree.ElementTree as ET
import arabic_reshaper
from bidi.algorithm import get_display

def display_arabic(text):
    # Reshape letters then apply bidi re-ordering for display
    reshaped = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped)
    return bidi_text

# ==============================================================================
# PART 1: BASIC DATA STRUCTURE
# ==============================================================================

class VerseData:
    """
    Simple container for verse information
    Think of this like a box that holds all info about one verse
    """
    
    def __init__(self, verse_id, surah_index, surah_name, verse_index, 
                 text, absolute_position, bismillah=None):
        # Basic identification
        self.verse_id = verse_id              # Example: "2:255"
        self.surah_index = surah_index        # Example: 2 (Al-Baqarah)
        self.surah_name = surah_name          # Example: "البقرة"
        self.verse_index = verse_index        # Example: 255
        
        # Content
        self.text = text                       # The actual Arabic text
        
        # Position tracking
        self.absolute_position = absolute_position  # Position 1-6236
        
        # Optional
        self.bismillah = bismillah            # Some verses have this


# ==============================================================================
# PART 2: MAIN LOOKUP SYSTEM
# ==============================================================================

class QuranLookupSystem:
    """
    A system to quickly find any verse in the Quran
    
    Think of it like a library catalog system:
    - You can search by book number (surah)
    - You can search by book name (surah name)
    - You can search by position on shelf (absolute position)
    """
    
    def __init__(self, xml_file_path):
        """
        Initialize the system by loading the Quran data
        
        Args:
            xml_file_path: Path to your quran.xml file
        """
        # Create empty dictionaries to store our data
        # Think of dictionaries like phone books - you look up a key to get a value
        
        self.verses = {}                    # Main storage: "2:255" → VerseData
        self.surah_index = {}               # Surah number → list of verse IDs
        self.surah_name_index = {}          # Arabic name → surah number
        self.absolute_position_index = {}   # 1-6236 → verse ID
        self.reverse_lookup = {}            # Verse ID → position number
        
        # Load all the data from XML file
        self._load_quran_from_xml(xml_file_path)
    
    
    def _load_quran_from_xml(self, xml_file_path):
        """
        Read the XML file and organize all verses
        The underscore _ at the start means "internal function - users don't need to call this"
        """
        # Parse the XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Track position as we go through verses
        absolute_pos = 0
        
        # Go through each surah
        for sura in root.findall('sura'):
            # Get surah information
            surah_num = int(sura.get('index'))        # Example: 1, 2, 3...
            surah_name = sura.get('name')              # Example: "الفاتحة"
            
            # Set up tracking for this surah
            self.surah_name_index[surah_name] = surah_num
            self.surah_index[surah_num] = []
            
            # Go through each verse (aya) in this surah
            for aya in sura.findall('aya'):
                verse_num = int(aya.get('index'))
                absolute_pos = absolute_pos + 1
                
                # Create a unique ID for this verse
                verse_id = str(surah_num) + ":" + str(verse_num)  # Example: "2:255"
                
                # Create a VerseData object with all the info
                verse_data = VerseData(
                    verse_id=verse_id,
                    surah_index=surah_num,
                    surah_name=surah_name,
                    verse_index=verse_num,
                    text=aya.get('text'),
                    absolute_position=absolute_pos,
                    bismillah=aya.get('bismillah')
                )
                
                # Store this verse in all our indexes for quick lookup
                self.verses[verse_id] = verse_data
                self.surah_index[surah_num].append(verse_id)
                self.absolute_position_index[absolute_pos] = verse_id
                self.reverse_lookup[verse_id] = absolute_pos
        
        print(f"✓ Loaded {len(self.verses)} verses from {len(self.surah_index)} surahs")
    
    
    # ==========================================================================
    # PART 3: LOOKUP METHODS (HOW TO FIND VERSES)
    # ==========================================================================
    
    def get_verse(self, surah, verse):
        """
        Get a verse by surah and verse number
        
        Example: get_verse(2, 255) returns Ayat al-Kursi
        
        Args:
            surah: Surah number (1-114)
            verse: Verse number within that surah
        
        Returns:
            VerseData object, or None if not found
        """
        verse_id = str(surah) + ":" + str(verse)
        
        # Look it up in our dictionary
        if verse_id in self.verses:
            return self.verses[verse_id]
        else:
            return None
    
    
    def get_verse_by_position(self, position):
        """
        Get a verse by its position in the entire Quran (1-6236)
        
        Example: get_verse_by_position(1) returns the first verse (Al-Fatiha 1:1)
        
        Args:
            position: Number from 1 to 6236
        
        Returns:
            VerseData object, or None if not found
        """
        # First find the verse_id at this position
        if position in self.absolute_position_index:
            verse_id = self.absolute_position_index[position]
            # Then get the actual verse data
            return self.verses[verse_id]
        else:
            return None
    
    
    def get_surah(self, surah_num):
        """
        Get all verses from one surah
        
        Example: get_surah(1) returns all 7 verses of Al-Fatiha
        
        Args:
            surah_num: Surah number (1-114)
        
        Returns:
            List of VerseData objects
        """
        if surah_num not in self.surah_index:
            return []
        
        # Get list of verse IDs in this surah
        verse_ids = self.surah_index[surah_num]
        
        # Convert each verse_id to actual VerseData
        verses = []
        for verse_id in verse_ids:
            verses.append(self.verses[verse_id])
        
        return verses
    
    
    def get_surah_by_name(self, surah_name):
        """
        Get all verses by the Arabic name of the surah
        
        Example: get_surah_by_name("الفاتحة") returns Al-Fatiha verses
        
        Args:
            surah_name: Arabic name like "الفاتحة" or "البقرة"
        
        Returns:
            List of VerseData objects
        """
        # First find which surah number this name corresponds to
        if surah_name not in self.surah_name_index:
            return []
        
        surah_num = self.surah_name_index[surah_name]
        
        # Now get all verses from that surah
        return self.get_surah(surah_num)
    
    
    def get_context(self, surah, verse, before=2, after=2):
        """
        Get a verse plus surrounding verses for context
        
        Example: get_context(2, 255, before=2, after=2)
        Returns the verse 2:255 plus 2 verses before and 2 after
        
        Args:
            surah: Surah number
            verse: Verse number
            before: How many verses before to include
            after: How many verses after to include
        
        Returns:
            Dictionary with keys "before", "target", "after"
        """
        # Get the target verse
        target_verse = self.get_verse(surah, verse)
        
        if target_verse is None:
            return {}
        
        # Find its position
        target_pos = target_verse.absolute_position
        
        # Get verses before
        before_verses = []
        for pos in range(target_pos - before, target_pos):
            if pos >= 1:  # Don't go below verse 1
                verse_data = self.get_verse_by_position(pos)
                if verse_data:
                    before_verses.append(verse_data)
        
        # Get verses after
        after_verses = []
        for pos in range(target_pos + 1, target_pos + after + 1):
            if pos <= 6236:  # Don't go beyond last verse
                verse_data = self.get_verse_by_position(pos)
                if verse_data:
                    after_verses.append(verse_data)
        
        return {
            "before": before_verses,
            "target": target_verse,
            "after": after_verses
        }


# ==============================================================================
# PART 4: USAGE EXAMPLES
# ==============================================================================

def simple_demo():
    """
    Simple examples showing how to use the system
    """
    print("="*60)
    print("SIMPLE QURAN LOOKUP EXAMPLES")
    print("="*60)
    
    # Initialize the system
    print("\nLoading Quran data...")
    quran = QuranLookupSystem('quran.xml')
    
    # Example 1: Get a specific verse
    print("\n" + "-"*60)
    print("Example 1: Get Ayat al-Kursi (Verse 2:255)")
    print("-"*60) 

    verse = quran.get_verse(2, 255)
    
    if verse:
        print(f"Surah: {verse.surah_name}")
        print(f"Location: {verse.surah_index}:{verse.verse_index}")
        print(f"Position in Quran: {verse.absolute_position}")
        print(f"Text: {verse.text}")
        #print(f"{display_arabic(verse.text)}")
    
    
    # Example 2: Get by position
    print("\n" + "-"*60)
    print("Example 2: Get the 100th verse in the Quran")
    print("-"*60) 
    
    verse = quran.get_verse_by_position(100)
    
    if verse:
        print(f"The 100th verse is: {verse.verse_id}")
        print(f"From Surah: {verse.surah_name}")
        print(f"Text: {verse.text}")
    
    
    # Example 3: Get entire surah
    print("\n" + "-"*60)
    print("Example 3: Get all verses of Al-Fatiha")
    print("-"*60)
    
    fatiha_verses = quran.get_surah(1)
    
    print(f"Al-Fatiha has {len(fatiha_verses)} verses:")
    for verse in fatiha_verses:
        print(f"  {verse.verse_index}: {verse.text}")
    
    
    # Example 4: Get with context
    print("\n" + "-"*60)
    print("Example 4: Get verse 2:255 with 2 verses before and after")
    print("-"*60)
    
    context = quran.get_context(2, 255, before=2, after=2)
    
    print("Verses BEFORE:")
    for v in context["before"]:
        print(f"  {v.verse_id}: {v.text[:50]}...")
    
    print("\nTARGET VERSE:")
    print(f"  {context['target'].verse_id}: {context['target'].text[:50]}...")
    
    print("\nVerses AFTER:")
    for v in context["after"]:
        print(f"  {v.verse_id}: {v.text[:50]}...")


# ==============================================================================
# RUN THE DEMO
# ==============================================================================

if __name__ == "__main__":
    simple_demo()


"""
EXPLANATION OF CONCEPTS:

1. DICTIONARIES {}
   Think of them like a phone book or address book
   Example: phone_book = {"John": "555-1234", "Jane": "555-5678"}
   To look up: phone_book["John"] gives you "555-1234"
   
   In our code:
   self.verses = {"1:1": VerseData(...), "2:255": VerseData(...)}
   Looking up: self.verses["2:255"] gives you that verse instantly

2. CLASSES
   A class is like a blueprint for creating objects
   Example: VerseData is a blueprint for storing verse information
   
   When you do: verse = VerseData(verse_id="2:255", ...)
   You're creating one specific instance with that data

3. self
   Inside a class, 'self' means "this particular instance"
   Example: self.verses means "this QuranLookupSystem's verse dictionary"
   It's how the object remembers its own data

4. def __init__
   This is the "constructor" - it runs when you create a new instance
   Example: quran = QuranLookupSystem('quran.xml')
   The __init__ function runs automatically to set everything up

5. UNDERSCORE _ PREFIX
   Functions starting with _ are "internal" or "private"
   Example: _load_quran_from_xml
   It means: "Users shouldn't call this directly, it's just for internal use"

6. None
   Python's way of saying "nothing" or "not found"
   Example: if verse is None: means "if no verse was found"

7. RETURN
   Sends data back to whoever called the function
   Example: return verse_data
   The caller can then use that data: my_verse = get_verse(2, 255)

NEXT STEPS FOR YOU:
1. Start with just the VerseData class - understand how it stores info
2. Then look at __init__ - see how data gets loaded
3. Try using get_verse() - the simplest lookup
4. Gradually add more complexity as you understand each part
"""