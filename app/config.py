import os
import torch


class Config:
    SECRET_KEY = '4adda9070acf5830ed45b76da140974147304941e9745a58'
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # MongoDB settings
    MONGODB_URI = 'mongodb://mongodb:27017/'
    MONGODB_DATABASE = 'geolocation_db'

    # MongoDB collection names (add these new settings)
    MONGODB_BINARY_COLLECTION = 'binary_codes'  # for your binary codes
    MONGODB_PATHS_COLLECTION = 'paths'  # for your paths
    MONGODB_DHN_COLLECTION = 'segmentation_dhn'  # for DHN segmentation data
    # MONGODB_HASHNET_COLLECTION = 'segmentation_hashnet'  # for HashNet segmentation data

    # Model paths
    MODEL_PATH = '/app/llava-v1.6-mistral-7b'

    # Geographical model paths and files
    GEO_MODEL_PATH = "./airbnb_14countries_train_database_128bits_0.6296825813840561/model.pt"
    DHN_MODEL_PATH = "./DHN_airbnb_14countries_512bits_0.3558918258844033/model.pt"
    # HASHNET_MODEL_PATH = "./Hashnet_airbnb_14countries_512bits_0.3702259939175427/model.pt"

    # Country and continent configurations
    COUNTRIES = ["Bolivia", "Germany", "Poland", "Pakistan", "Chile",
                 "Kazakhstan", "Japan", "Argentina", "Colombia",
                 "Norway", "France", "Peru", "Hungary", "KoreaSouth", "Portugal"]

    CONTINENTS_DICT = {
        "Europe": ["Germany", "Poland", "Norway", "France", "Hungary", "Portugal"],
        "Asia": ["Pakistan", "Kazakhstan", "Japan", "KoreaSouth"],
        "latinAmerica": ["Bolivia", "Chile", "Argentina", "Colombia", "Peru"]
    }

    # API endpoints
    MATERIAL_RECOGNITION_URL = 'http://materobot:5001/material_recognition'

    # COUNTRY_NEIGHBORS = {
    #     # Latin American countries
    #     "Argentina": ["Chile", "Peru", "Bolivia", "Colombia"],
    #     "Bolivia": ["Peru", "Chile", "Argentina", "Colombia"],
    #     "Chile": ["Argentina", "Peru", "Bolivia"],
    #     "Colombia": ["Peru", "Bolivia", "Argentina"],
    #     "Peru": ["Bolivia", "Chile", "Colombia", "Argentina"],
    #
    #     # European countries
    #     "France": ["Germany", "Poland", "Hungary"],
    #     "Germany": ["France", "Poland", "Hungary"],
    #     "Hungary": ["Poland", "Germany", "France"],
    #     "Norway": ["Germany", "Poland"],
    #     "Poland": ["Germany", "Hungary", "France"],
    #
    #     # Asian countries
    #     "Japan": ["KoreaSouth"],
    #     "KoreaSouth": ["Japan"],
    #     "Kazakhstan": ["Pakistan", "KoreaSouth"],
    #     "Pakistan": ["Kazakhstan"]
    # }

 # Add the country descriptions dictionary chatgpt and claude
    COUNTRY_DESCRIPTIONS = {
        "Bolivia": "Bolivian interiors reflect a rich mix of indigenous culture and Spanish colonial influence. Homes often incorporate vibrant, traditional Andean textiles with intricate patterns and bright colors, used as wall hangings, rugs, or cushion covers. Natural materials like wood and stone are prominent, providing a rustic feel. Woven baskets, handcrafted pottery, and carved wooden furniture are common decorative elements. Warm, earthy tones dominate, reflecting the country's mountainous landscape. Bolivians often use decorative arches and niches in walls to display artifacts, emphasizing a sense of heritage and craftsmanship. Traditional aguayos (woven textiles) are frequently used both decoratively and functionally. Adobe construction remains common in traditional homes, providing natural insulation.",

        "Germany": "German interiors are characterized by minimalism and functionality. Neutral color schemes, clean lines, and uncluttered spaces create a calm, organized environment. Wood, particularly light varieties like oak and pine, is a common element, paired with stone or ceramic for a touch of contrast. Integrated storage solutions keep homes tidy and efficient. Large windows bring in natural light, often complemented by sheer curtains or minimal coverings. Indoor plants are popular for adding a touch of nature. Decor is carefully chosen, with a focus on quality over quantity, often including modern art and practical furnishings. The Bauhaus influence remains strong, with an emphasis on form following function. Energy efficiency is a key consideration in modern German homes, with advanced heating systems and well-insulated windows being standard features.",

        "Poland": "Polish interiors blend traditional elements with contemporary design. Classic wooden furniture, often featuring intricate carving or rustic finishes, is juxtaposed with modern minimalist pieces. Neutral colors like whites and grays dominate, accented by warm woods and pops of color from textiles or artwork. Decor often includes floral motifs and traditional Polish ceramics, such as Bolesławiec pottery. Homes may feature parquet flooring and woven rugs. Polish interiors balance coziness and practicality, emphasizing family heirlooms and cultural artifacts. Traditional kilims (flat-woven rugs) are often used as both floor coverings and wall decorations. Religious iconography, particularly in Catholic households, is commonly displayed.",

        "Pakistan": "Pakistani homes showcase a blend of traditional Islamic architecture and modern influences. Key features include ornate woodwork, carved furniture, and intricately designed metalwork. Fabrics with rich embroidery, such as Kashmiri or Sindhi patterns, are used for curtains, cushions, and wall hangings. Warm colors like deep reds, oranges, and golds are prevalent, alongside neutral bases. Hand-knotted rugs, marble inlays, and decorative tiles with Islamic geometric patterns are often present. Indoor spaces may have arches or niches that hold decorative items, and traditional lamps and lanterns provide ambient lighting. Courtyards (haveli style) are common in traditional homes, providing natural ventilation and a private outdoor space. Calligraphy featuring Quranic verses is often incorporated into the decor.",

        "Chile": "Chilean interiors often reflect a combination of colonial and modern styles. Homes incorporate natural materials like wood and stone to create a warm, inviting atmosphere. Rustic wooden beams, parquet or stone flooring, and cozy textiles such as wool or alpaca throws are common. Earthy colors, complemented by whites and natural greens, reflect Chile's diverse landscapes. Decorative touches often include handmade pottery, woven baskets, and artworks that showcase local craftsmanship. Large windows or glass doors are popular for inviting natural light and showcasing outdoor views. In coastal areas, homes often feature nautical themes and lighter colors, while mountain homes tend toward heavier textiles and darker woods.",

        "Kazakhstan": "Kazakh interiors blend traditional nomadic elements with modern design. Rich textiles featuring ethnic patterns, such as shyrdaks (felt rugs) and decorative tapestries, bring color and warmth. Ornate wooden furniture, often carved with traditional motifs, is common, as are decorative items made from metals like copper and brass. Neutral walls are accentuated with deep reds, blues, and golds. Homes often display symbolic items such as eagle or horse motifs, reflecting the country's nomadic heritage. Large windows and modern lighting add a contemporary feel to the traditional decor. Traditional tuskiiz (wall hangings) with embroidered patterns are still commonly used in modern homes.",

        "Japan": "Japanese interiors epitomize minimalism and harmony with nature. Spaces are often open and uncluttered, featuring both traditional sliding doors (fusuma or shoji) and modern glass walls. Textiles add warmth through noren curtains, zabuton cushions, and futon bedding. Natural materials like wood (cedar, cypress) and bamboo are essential, alongside tatami mat flooring. Neutral colors dominate, with earthy tones and subtle accent colors. The design emphasizes clean lines and the integration of nature through indoor plants or elements like rock gardens and water features. Furniture is low to the ground, and decor is minimal, focusing on simplicity and functionality. Tokonoma (alcoves) are traditional features used to display seasonal art or ikebana. Storage is often built into walls, with traditional tansu chests serving both practical and decorative purposes.",

        "Argentina": "Argentinian homes often combine European elegance with Latin American warmth. Interiors feature high-quality wood flooring, ornate furniture with European influences, and decorative items like traditional woven fabrics and gaucho-inspired elements. Bright, colorful textiles and hand-painted ceramics add vibrancy to neutral backgrounds. Large windows or glass doors are common to let in ample natural light. Leather and cowhide rugs are popular, reflecting Argentina's strong ties to cattle ranching. Artworks, including paintings and sculptures, often highlight local talent and history. Mate gourds and bombillas are often displayed as both functional items and decorative elements.",

        "Colombia": "Colombian interiors are a mix of Spanish colonial architecture and indigenous influences. Decorative elements include colorful textiles, hand-woven baskets, and vibrant wall art. Wood is extensively used, particularly in furniture and beams, paired with stone or ceramic tiles for flooring. Brightly colored walls or accent pieces in blues, reds, and yellows echo the country's natural and cultural vibrancy. Indoor plants, including tropical varieties like ferns and orchids, are frequently used to bring nature inside. Homes are designed to be welcoming, emphasizing comfort and a connection to the outdoors. Hammocks are often incorporated as both functional and decorative elements. Mochila bags, traditional Colombian handicrafts, are frequently displayed as wall decorations.",

        "Norway": "Norwegian interiors follow the principles of Scandinavian design—minimalist, functional, and cozy (hygge). Light, neutral color palettes of white, beige, and gray are paired with natural wood and stone. Spaces are kept simple and uncluttered, with an emphasis on high-quality, sustainable materials. Large windows allow ample natural light to fill the rooms, often left undressed or minimally covered. Furniture is sleek and functional, with soft textures from wool or sheepskin throws adding warmth. Decor is minimal but includes touches like candles, art pieces, and indoor plants to create a homely atmosphere. Wood-burning stoves (peisovn) are common features, serving both practical and aesthetic purposes. Traditional Norwegian rosemaling (decorative painting) occasionally appears in more traditional homes.",

        "France": "French interiors blend elegance with rustic charm, particularly evident in the mix of Haussmannian apartments and countryside homes. Richly detailed moldings, parquet floors, and large windows are typical features. Furniture often includes antique or vintage pieces with an ornate touch, while modern design elements provide a contrast. Neutral colors like white, cream, and light gray are popular, accented with splashes of color through artwork or textiles. Decor includes mirrors with gilded frames, chandeliers, and classic French ceramics or porcelain. French homes focus on sophistication and timeless appeal. Herringbone parquet flooring (point de Hongrie) is a classic feature in traditional apartments. Trumeau mirrors above fireplaces are characteristic of traditional French interiors.",

        "Peru": "Peruvian homes are deeply influenced by indigenous and colonial traditions. Interiors feature vibrant textiles with intricate geometric patterns, woven from natural fibers like alpaca wool. Wood and stone are commonly used for flooring and furniture, often complemented by handmade pottery and artisan crafts. Bright, warm colors such as orange, red, and yellow create a lively atmosphere, balanced by more neutral walls. Homes often incorporate decorative elements that reflect the Andean landscape, including framed art or murals. The design is rustic yet rich in cultural symbolism. Traditional retablos (portable religious altarpieces) are common decorative elements. Ceramic toritos de Pucará (Pucará bulls) are often displayed as symbols of protection and good fortune.",

        "Hungary": "Hungarian interiors combine historic elements with modern touches. Traditional folk art, such as Matyó embroidery with its vivid floral patterns, adds color and character to homes. Furniture often features carved wood with classic, detailed designs. Decorative tiles, especially those inspired by Zsolnay ceramics, are popular in kitchens and fireplaces. Neutral backgrounds like white or beige are accented with pops of bright reds, greens, or blues. Hungarian interiors prioritize comfort, incorporating plush textiles and family heirlooms that connect to the country's rich heritage. Traditional ceramic stoves (cserépkályha) are still found in many homes, serving both functional and decorative purposes.",

        "KoreaSouth": "South Korean interiors blend traditional elements like hanok architecture with modern minimalism. While traditional homes feature wooden beams and paper-covered sliding doors (jangji), contemporary spaces incorporate glass walls and partitions. Traditional textiles remain important, from bojagi fabric art to floor cushions and decorative tapestries. Light, neutral colors and simple lines create a tranquil, uncluttered atmosphere. Floor seating and low tables remain popular, reflecting traditional Korean culture. Modern homes incorporate sleek, functional furniture, often with built-in storage. Indoor plants and minimalist decor create an inviting environment, with an emphasis on natural light and open spaces. Ondol (traditional floor heating systems) remain a distinctive feature in modern Korean homes. Traditional moon jars (dal hang-ari) are often used as decorative elements.",

        "Portugal": "Residential structures in Portugal often exhibit whitewashed walls, local stone (such as granite in the north or schist in central regions), and terracotta roof tiles. Distinctive regional variations exist, from the robust Casa Minhota with thick granite walls in the north to the colorful wooden houses along coastal areas. A defining feature is the presence of azulejo tiles-painted tin-glazed ceramic tiles-frequently used on exterior facades and interior walls, displaying blue and white or colorful patterns that often depict historical scenes or natural motifs. In the Algarve, ornate chimneys and decorative parapet walls (platibandas) serve as status symbols. Traditional interiors feature dark, carved wooden furniture, terracotta or wood flooring, and colorful handcrafted textiles like Arraiolos rugs. Common decorative elements include rooster and swallow figures, alongside various handmade ceramics depicting everyday life. Modern designs show clean lines and open-plan layouts but typically integrate traditional elements like azulejos, natural materials, and a warm color palette of whites, blues, terracottas, and earthy tones. A strong connection to the outdoors remains essential, with terraces (açoteias), patios, balconies, or courtyards that reflect Portugal's Mediterranean lifestyle and Moorish influences."
    }