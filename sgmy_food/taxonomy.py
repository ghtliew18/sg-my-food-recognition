"""
Food Taxonomy Module
====================
Contains the complete taxonomy of 50 Singapore & Malaysian foods.
"""

from typing import List, Dict

# Simple list of food labels for quick reference
SG_MY_FOOD_LABELS: List[str] = [
    "Nasi Lemak", "Chicken Rice", "Laksa", "Char Kway Teow", "Hokkien Mee",
    "Bak Kut Teh", "Roti Prata", "Mee Goreng", "Satay", "Rojak",
    "Chilli Crab", "Black Pepper Crab", "Hainanese Chicken Rice",
    "Wonton Mee", "Popiah", "Kueh Pie Tee", "Curry Puff", "Tau Huay",
    "Ice Kacang", "Cendol", "Nasi Kandar", "Nasi Dagang",
    "Rendang", "Asam Pedas", "Ikan Bakar", "Mee Rebus", "Lontong",
    "Otak-Otak", "Kuih Lapis", "Ondeh-Ondeh", "Pandan Cake",
    "Teh Tarik", "Kaya Toast", "Soft Boiled Eggs", "Nasi Goreng",
    "Murtabak", "Banana Leaf Rice", "Curry Laksa", "Assam Laksa",
    "Prawn Mee", "Lor Mee", "Yong Tau Foo", "Duck Rice",
    "Braised Pork Rice", "Claypot Rice", "Economy Rice",
    "Bak Chor Mee", "Carrot Cake", "Fish Head Curry", "Mee Siam",
]

# Set for O(1) lookup
SG_MY_FOOD_SET = set(label.lower() for label in SG_MY_FOOD_LABELS)

# Full taxonomy with metadata for dataset generation
FOOD_TAXONOMY: List[Dict] = [
    # Rice Dishes
    {"label": "Nasi Lemak", "search_terms": ["nasi lemak food", "nasi lemak malaysia dish"], "cuisine_region": "Malaysia", "description": "Fragrant coconut rice served with sambal, anchovies, peanuts, and egg."},
    {"label": "Chicken Rice", "search_terms": ["hainanese chicken rice dish", "singapore chicken rice food"], "cuisine_region": "Singapore", "description": "Poached chicken served with fragrant rice cooked in chicken broth."},
    {"label": "Nasi Goreng", "search_terms": ["nasi goreng fried rice", "indonesian nasi goreng"], "cuisine_region": "Both", "description": "Wok-fried rice with sweet soy sauce and toppings."},
    {"label": "Nasi Kandar", "search_terms": ["nasi kandar penang", "nasi kandar food"], "cuisine_region": "Malaysia", "description": "Steamed rice served with various curries."},
    {"label": "Nasi Dagang", "search_terms": ["nasi dagang terengganu", "nasi dagang fish curry"], "cuisine_region": "Malaysia", "description": "Rice steamed in coconut milk with fish curry."},
    {"label": "Claypot Rice", "search_terms": ["claypot chicken rice", "clay pot rice chinese"], "cuisine_region": "Both", "description": "Rice cooked in claypot with chicken and Chinese sausage."},
    {"label": "Economy Rice", "search_terms": ["economy rice singapore", "chap fan mixed rice"], "cuisine_region": "Both", "description": "Buffet style rice with various dishes."},
    {"label": "Duck Rice", "search_terms": ["braised duck rice teochew", "roast duck rice"], "cuisine_region": "Singapore", "description": "Braised duck served over rice."},
    {"label": "Braised Pork Rice", "search_terms": ["lu rou fan braised pork", "taiwanese braised pork rice"], "cuisine_region": "Both", "description": "Minced pork braised in soy sauce over rice."},
    {"label": "Banana Leaf Rice", "search_terms": ["banana leaf rice indian", "banana leaf rice malaysia"], "cuisine_region": "Malaysia", "description": "Rice on banana leaf with curries."},
    
    # Noodle Dishes
    {"label": "Laksa", "search_terms": ["laksa noodle soup", "curry laksa singapore"], "cuisine_region": "Both", "description": "Spicy coconut milk noodle soup."},
    {"label": "Char Kway Teow", "search_terms": ["char kway teow noodles", "penang char kuey teow"], "cuisine_region": "Both", "description": "Stir-fried flat rice noodles."},
    {"label": "Hokkien Mee", "search_terms": ["hokkien mee prawn noodles", "singapore hokkien mee"], "cuisine_region": "Singapore", "description": "Stir-fried noodles in prawn stock."},
    {"label": "Mee Goreng", "search_terms": ["mee goreng mamak noodles", "indian mee goreng"], "cuisine_region": "Both", "description": "Spicy fried noodles."},
    {"label": "Wonton Mee", "search_terms": ["wonton noodles dumpling", "wanton mee hong kong"], "cuisine_region": "Both", "description": "Egg noodles with pork dumplings."},
    {"label": "Prawn Mee", "search_terms": ["prawn mee soup", "har mee penang"], "cuisine_region": "Both", "description": "Noodles in prawn broth."},
    {"label": "Lor Mee", "search_terms": ["lor mee singapore", "braised noodles gravy"], "cuisine_region": "Singapore", "description": "Noodles in thick starchy gravy."},
    {"label": "Mee Rebus", "search_terms": ["mee rebus gravy noodles", "mee rebus malay"], "cuisine_region": "Both", "description": "Noodles in sweet potato gravy."},
    {"label": "Curry Laksa", "search_terms": ["curry laksa coconut", "curry mee malaysia"], "cuisine_region": "Malaysia", "description": "Creamy curry coconut noodle soup."},
    {"label": "Assam Laksa", "search_terms": ["assam laksa penang sour", "asam laksa fish"], "cuisine_region": "Malaysia", "description": "Sour fish-based noodle soup."},
    
    # Meat & Seafood
    {"label": "Satay", "search_terms": ["satay skewers peanut sauce", "chicken satay grilled"], "cuisine_region": "Both", "description": "Grilled meat skewers with peanut sauce."},
    {"label": "Rendang", "search_terms": ["beef rendang dry curry", "rendang padang indonesian"], "cuisine_region": "Both", "description": "Slow-cooked dry curry beef."},
    {"label": "Chilli Crab", "search_terms": ["singapore chilli crab seafood", "chili crab dish"], "cuisine_region": "Singapore", "description": "Crab in tomato-chili sauce."},
    {"label": "Black Pepper Crab", "search_terms": ["black pepper crab singapore", "pepper crab wok"], "cuisine_region": "Singapore", "description": "Crab with black pepper."},
    {"label": "Bak Kut Teh", "search_terms": ["bak kut teh pork ribs soup", "herbal pork soup"], "cuisine_region": "Both", "description": "Pork rib soup with herbs."},
    {"label": "Asam Pedas", "search_terms": ["asam pedas sour spicy fish", "asam pedas melaka"], "cuisine_region": "Malaysia", "description": "Sour spicy fish stew."},
    {"label": "Ikan Bakar", "search_terms": ["ikan bakar grilled fish", "grilled fish sambal"], "cuisine_region": "Malaysia", "description": "Grilled fish in banana leaf."},
    {"label": "Otak-Otak", "search_terms": ["otak otak grilled fish cake", "otah otah"], "cuisine_region": "Both", "description": "Grilled fish paste."},
    
    # Bread & Snacks
    {"label": "Roti Prata", "search_terms": ["roti prata flatbread", "roti canai crispy"], "cuisine_region": "Both", "description": "Crispy layered flatbread."},
    {"label": "Murtabak", "search_terms": ["murtabak stuffed pancake", "martabak meat"], "cuisine_region": "Both", "description": "Stuffed pancake with meat."},
    {"label": "Popiah", "search_terms": ["popiah fresh spring roll", "popiah wrapper"], "cuisine_region": "Both", "description": "Fresh spring roll."},
    {"label": "Kueh Pie Tee", "search_terms": ["kueh pie tee top hats", "pie tee crispy cups"], "cuisine_region": "Both", "description": "Crispy pastry cups."},
    {"label": "Curry Puff", "search_terms": ["curry puff pastry", "karipap spiral"], "cuisine_region": "Both", "description": "Fried pastry with curry filling."},
    {"label": "Yong Tau Foo", "search_terms": ["yong tau foo stuffed tofu", "hakka yong tau fu"], "cuisine_region": "Both", "description": "Stuffed tofu and vegetables."},
    
    # Breakfast & Drinks
    {"label": "Kaya Toast", "search_terms": ["kaya toast coconut jam", "singapore breakfast kaya"], "cuisine_region": "Singapore", "description": "Toast with coconut jam."},
    {"label": "Soft Boiled Eggs", "search_terms": ["soft boiled eggs soy sauce", "kopitiam eggs singapore"], "cuisine_region": "Singapore", "description": "Soft eggs with soy sauce."},
    {"label": "Teh Tarik", "search_terms": ["teh tarik pulled tea", "malaysian milk tea"], "cuisine_region": "Malaysia", "description": "Pulled milk tea."},
    
    # Rojak & Salads
    {"label": "Rojak", "search_terms": ["rojak fruit salad shrimp paste", "penang rojak"], "cuisine_region": "Both", "description": "Mixed salad with shrimp paste."},
    {"label": "Lontong", "search_terms": ["lontong sayur rice cake", "lontong curry"], "cuisine_region": "Both", "description": "Rice cakes in coconut curry."},
    
    # Desserts
    {"label": "Tau Huay", "search_terms": ["tau huay soybean pudding", "douhua dessert"], "cuisine_region": "Both", "description": "Soybean pudding."},
    {"label": "Ice Kacang", "search_terms": ["ice kacang shaved ice", "ais kacang dessert"], "cuisine_region": "Both", "description": "Shaved ice dessert."},
    {"label": "Cendol", "search_terms": ["cendol green jelly dessert", "chendol coconut"], "cuisine_region": "Both", "description": "Shaved ice with green jelly."},
    {"label": "Kuih Lapis", "search_terms": ["kuih lapis layer cake", "kue lapis steamed"], "cuisine_region": "Both", "description": "Colorful layered cake."},
    {"label": "Ondeh-Ondeh", "search_terms": ["ondeh ondeh pandan balls", "onde onde gula melaka"], "cuisine_region": "Both", "description": "Glutinous rice balls with palm sugar."},
    {"label": "Pandan Cake", "search_terms": ["pandan chiffon cake green", "pandan layer cake"], "cuisine_region": "Both", "description": "Pandan flavored cake."},
    
    # Additional
    {"label": "Hainanese Chicken Rice", "search_terms": ["hainanese chicken rice ball", "melaka chicken rice"], "cuisine_region": "Both", "description": "Signature dish of poached chicken with aromatic rice."},
    {"label": "Bak Chor Mee", "search_terms": ["bak chor mee minced pork noodles", "bcm singapore"], "cuisine_region": "Singapore", "description": "Noodles with minced pork."},
    {"label": "Carrot Cake", "search_terms": ["chai tow kway fried carrot cake", "radish cake singapore"], "cuisine_region": "Singapore", "description": "Pan-fried radish cake."},
    {"label": "Fish Head Curry", "search_terms": ["fish head curry singapore indian", "curry fish head"], "cuisine_region": "Singapore", "description": "Fish head in curry."},
    {"label": "Mee Siam", "search_terms": ["mee siam vermicelli", "mee siam gravy"], "cuisine_region": "Both", "description": "Rice vermicelli in tangy gravy."},
]


def get_food_by_label(label: str) -> Dict:
    """Get food metadata by label."""
    for food in FOOD_TAXONOMY:
        if food["label"].lower() == label.lower():
            return food
    return None


def get_foods_by_region(region: str) -> List[Dict]:
    """Get all foods from a specific region."""
    return [f for f in FOOD_TAXONOMY if f["cuisine_region"] == region or f["cuisine_region"] == "Both"]


def is_sg_my_food(label: str) -> bool:
    """Check if a label is a known SG/MY food."""
    return label.lower() in SG_MY_FOOD_SET
