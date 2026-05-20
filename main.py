import argparse
import concurrent.futures
import os
import re
import shutil
import time
from urllib.parse import urldefrag, urlparse

import chromadb
import ollama
import requests
from bs4 import BeautifulSoup


#CONFIGURATION
ALLOWED_DOMAIN = "www.uia.no"                       #Allowed website domain

CHROMA_FOLDER = "uia_chroma_db"                     #Folder for vector database
COLLECTION_NAME = "uia_ikt_courses"                 #Name of ChromaDB collection

EMBED_MODEL = "nomic-embed-text"                   
LLM_MODEL = "llama3.2"                             

TOP_K = 12                                          #Max retrieved chunks
MAX_SEMANTIC_DISTANCE = 0.44
MAX_CHUNKS_PER_COURSE = 2                           #Max chunks per course in broad search
CHUNK_SIZE = 900                                    #Characters per text chunk
CHUNK_OVERLAP = 150                                 #Overlap between chunks

YEARS = [2026, 2025]                                #Course years to check
SEASONS = ["spring", "autumn"]                      #Semesters to check

MIN_COURSE_NUMBER = 1                               #Lowest course number to generate
MAX_COURSE_NUMBER = 899                             #Highest course number to generate

HEADERS = {                                         #HTTP headers for requests
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

COURSE_CODE_RE = re.compile(                        #Matches IKT211, IKT 211, IKT-211
    r"\bIKT\s*-?\s*\d{3,4}[A-Z]?\b",
    re.IGNORECASE,
)



#UTILITY
def normalize_course_code(raw_code):
    return re.sub(r"[\s-]+", "", raw_code).upper()  #Remove spaces and hyphens


def extract_course_codes_from_question(question):
    matches = COURSE_CODE_RE.findall(question)       #Find all course codes in question
    return list(dict.fromkeys(normalize_course_code(match) for match in matches)) #Remove duplicates


def normalize_url(url):
    url, _ = urldefrag(url)                          #Remove URL fragment
    return url.rstrip("/")                           #Remove trailing slash


def is_uia_url(url):
    parsed = urlparse(url)                           #Parse URL into parts
    return parsed.scheme in {"http", "https"} and parsed.netloc == ALLOWED_DOMAIN #Check domain



#URL DISCOVERY
def extract_code_from_url(url):
    match = re.search(r"/(ikt\d{3,4}[a-z]?)\.html$", url.lower()) #Find code in URL
    return match.group(1).upper() if match else None #Return code if found


def extract_term_from_url(url):
    match = re.search(                               #Find year and semester in URL
        r"/english/studies/courses/(\d{4})/(spring|autumn)/",
        url.lower(),
    )

    if not match:
        return "unknown"                             #Return unknown if URL pattern fails

    year = match.group(1)                            #Extract year
    season = match.group(2)                          #Extract semester

    return f"{season.capitalize()} {year}"           #Return readable term


def build_candidate_course_urls():
    urls = []                                        #Store generated URLs

    for year in YEARS:                               #Loop through years
        for season in SEASONS:                       #Loop through semesters
            for number in range(MIN_COURSE_NUMBER, MAX_COURSE_NUMBER + 1): #Loop through numbers
                code = f"ikt{number:03d}"            #Create course code
                urls.append(                         #Add possible UiA course URL
                    f"https://www.uia.no/english/studies/courses/{year}/{season}/{code}.html"
                )

    return urls                                      #Return all candidate URLs


def url_exists(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=8, stream=True) #Request page
        content_type = response.headers.get("content-type", "").lower() #Read content type

        return response.status_code == 200 and "text/html" in content_type #Keep only HTML pages

    except requests.RequestException:
        return False                                 #Ignore failed requests


def choose_best_url(urls):
    def score(url):
        term = extract_term_from_url(url).lower()    #Get term from URL

        year_score = 2 if "2026" in term else 1 if "2025" in term else 0 #Prefer newest year
        season_score = 1 if "autumn" in term else 0  #Prefer autumn if possible

        return year_score, season_score              #Return sorting score

    return sorted(urls, key=score, reverse=True)[0]  #Return best URL


def discover_ikt_course_urls(max_workers=20):
    candidates = build_candidate_course_urls()       #Generate possible URLs
    found_by_code = {}                               #Store found URLs by course code

    print("Discovering UiA IKT course URLs...")
    print(f"Candidate URLs: {len(candidates)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor: #Parallel checking
        future_to_url = {
            executor.submit(url_exists, url): url
            for url in candidates
        }

        checked = 0                                  #Count checked URLs

        for future in concurrent.futures.as_completed(future_to_url): #Handle completed checks
            url = future_to_url[future]              #Get URL for this future
            checked += 1                             #Increase counter

            try:
                exists = future.result()             #Get result from thread
            except Exception:
                exists = False                       #Treat errors as not found

            if exists:
                course_code = extract_code_from_url(url) #Extract course code

                if course_code:
                    found_by_code.setdefault(course_code, []).append(url) #Store URL under code

            if checked % 250 == 0:
                print(f"Checked {checked}/{len(candidates)} URLs...") #Show progress

    discovered = []                                  #Final course list

    for course_code, urls in sorted(found_by_code.items()): #Loop through discovered courses
        best_url = choose_best_url(urls)             #Choose representative URL
        terms = sorted(extract_term_from_url(url) for url in urls) #Collect found terms

        discovered.append(
            {
                "course_code": course_code,
                "url": best_url,
                "all_terms": ", ".join(terms),
                "all_urls": " | ".join(sorted(urls)),
            }
        )

    print(f"Discovered {len(discovered)} unique IKT courses.")
    return discovered                                #Return unique courses



#SCRAPING AND CLEANING
def fetch_html(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15) #Fetch page

        if response.status_code != 200:
            return None                                 #Skip failed pages

        content_type = response.headers.get("content-type", "").lower() #Check content type

        if "text/html" not in content_type:
            return None                                 #Skip non-HTML content

        return response.text                            #Return raw HTML

    except requests.RequestException:
        return None                                     #Return None on request error


def clean_text(text):
    text = text.replace("\xa0", " ")                    #Replace non-breaking spaces
    text = re.sub(r"\s+", " ", text)                    #Normalize whitespace

    noise = [                                           #Common website text to remove
        "Skip to main content",
        "University of Agder",
        "Menu",
        "Search",
        "Breadcrumb",
        "Table of contents",
    ]

    for phrase in noise:
        text = text.replace(phrase, " ")                #Remove noisy phrase

    return re.sub(r"\s+", " ", text).strip()            #Return cleaned text


def extract_title_and_text(html):
    soup = BeautifulSoup(html, "html.parser")           #Parse HTML

    for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "svg"]):
        tag.decompose()                                 #Remove non-content elements

    title = soup.title.get_text(" ", strip=True) if soup.title else "No title" #Extract title

    main = soup.find("main")                            #Prefer main page content
    if main:
        text = main.get_text(" ", strip=True)           #Extract main text
    else:
        text = soup.get_text(" ", strip=True)           #Fallback to full page text

    return clean_text(title), clean_text(text)          #Return clean title and text


def extract_course_title(title, course_code):
    title = clean_text(title)                           #Clean page title

    title = re.sub(r"\s+-\s+Universitetet i Agder.*$", "", title) #Remove UiA suffix
    title = re.sub(r"\s+-\s*$", "", title)              #Remove trailing dash

    if "|" in title:
        parts = [part.strip() for part in title.split("|")] #Split title parts
        for part in parts:
            if course_code in part.upper():             #Find part containing code
                return part

    return title                                        #Return cleaned title


#STRUCTURED FIELD EXTRACTION

def extract_field(text, field_names):
    for field in field_names:                           #Try each possible field label
        pattern = (
            rf"{re.escape(field)}\s*[:\-]?\s*"
            rf"(.+?)(?=\s+[A-ZÆØÅ][A-Za-zÆØÅæøå /\-()]+?\s*[:\-]?\s|$)"
        )

        match = re.search(pattern, text, re.IGNORECASE) #Search for field value

        if match:
            value = match.group(1).strip()              #Extract value
            value = re.sub(r"\s+", " ", value)          #Normalize spacing
            return value[:300]                          #Limit long values

    return "Unknown"                                    #Fallback value


def extract_yes_no_field(text, field_names):
    for field in field_names:                           #Try each field label
        patterns = [
            rf"{re.escape(field)}\s*[:\-]?\s*(Yes|No)\b",
            rf"{re.escape(field)}\s*[:\-]?\s*(Ja|Nei)\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE) #Find yes/no value

            if match:
                value = match.group(1).lower()          #Normalize value

                if value in {"yes", "ja"}:
                    return "Yes"                        #Return English yes

                if value in {"no", "nei"}:
                    return "No"                         #Return English no

    return "Unknown"                                    #Fallback value


def extract_free_standing_status(text):
    return extract_yes_no_field(                        #Extract free-standing status
        text,
        [
            "Free-standing course",
            "Free standing course",
            "Offered as a free-standing course",
            "Offered as free-standing course",
            "Frittstående emne",
        ],
    )


def extract_credits(text):
    patterns = [                                         #Possible credit labels
        r"Credits\s*[:\-]?\s*([0-9]+(?:[.,][0-9]+)?)",
        r"Studiepoeng\s*[:\-]?\s*([0-9]+(?:[.,][0-9]+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)  #Search for credits

        if match:
            return match.group(1).replace(",", ".")      #Normalize decimal symbol

    return "Unknown"                                     #Fallback value


def extract_assessment(text):
    field_names = [                                      #Possible assessment labels
        "Assessment methods and criteria",
        "Assessment methods",
        "Examination",
        "Eksamen",
        "Vurderingsform",
    ]

    for field in field_names:
        idx = text.lower().find(field.lower())           #Find field position

        if idx != -1:
            snippet = text[idx:idx + 1200]               #Take nearby text
            return clean_text(snippet)                   #Return clean snippet

    return "Unknown"                                     #Fallback value


def extract_learning_outcomes(text):
    field_names = [                                      #Possible learning outcome labels
        "Learning outcomes",
        "Læringsutbytte",
    ]

    for field in field_names:
        idx = text.lower().find(field.lower())           #Find field position

        if idx != -1:
            snippet = text[idx:idx + 1200]               #Take nearby text
            return clean_text(snippet)                   #Return clean snippet

    return "Unknown"                                     #Fallback value


def extract_course_structured_data(course_code, course_title, url, all_terms, all_urls, text):
    return {                                             #Collect metadata for one course
        "course_code": course_code,
        "title": course_title,
        "url": url,
        "all_urls": all_urls,
        "all_terms": all_terms,
        "free_standing": extract_free_standing_status(text),
        "credits": extract_credits(text),
        "assessment": extract_assessment(text),
        "learning_outcomes": extract_learning_outcomes(text),
    }



#CHUNKING
def split_into_chunks(text):
    chunks = []                                          #Store text chunks
    start = 0                                            #Start index

    while start < len(text):
        end = start + CHUNK_SIZE                         #End index for chunk
        chunk = text[start:end].strip()                  #Extract chunk

        if len(chunk) > 120:
            chunks.append(chunk)                         #Keep only useful chunks

        start += CHUNK_SIZE - CHUNK_OVERLAP              #Move forward with overlap

    return chunks                                        #Return chunks



#EMBEDDINGS AND VECTOR DATABASE
def get_embedding(text):
    response = ollama.embed(model=EMBED_MODEL, input=text) #Create embedding with Ollama
    return response["embeddings"][0]                     #Return first embedding


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_FOLDER) #Open persistent ChromaDB client

    return client.get_or_create_collection(              #Get or create vector collection
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_database():
    if os.path.exists(CHROMA_FOLDER):
        shutil.rmtree(CHROMA_FOLDER)                     #Delete old vector database
        print(f"Deleted old vector database: {CHROMA_FOLDER}")


#CRAWL AND INDEX

def crawl_and_index(max_pages):
    collection = get_collection()                        #Open vector collection
    discovered_courses = discover_ikt_course_urls()      #Find IKT course pages

    if max_pages:
        discovered_courses = discovered_courses[:max_pages] #Limit number of pages

    if not discovered_courses:
        print("No IKT course URLs were discovered.")
        return                                           #Stop if no courses found

    total_chunks = 0                                     #Count indexed chunks
    indexed_pages = 0                                    #Count indexed pages

    print("\nIndexing discovered unique IKT course pages...")

    for page_index, course in enumerate(discovered_courses, start=1):
        course_code = course["course_code"]              #Get course code
        url = course["url"]                              #Get representative URL
        all_terms = course["all_terms"]                  #Get all found terms
        all_urls = course["all_urls"]                    #Get all found URLs

        print(f"\nIndexing [{page_index}/{len(discovered_courses)}]: {course_code}")
        print(url)

        html = fetch_html(url)                           #Fetch course page

        if not html:
            print("Could not fetch page, skipping.")
            continue                                     #Skip failed page

        title, text = extract_title_and_text(html)       #Extract clean text

        if len(text) < 200:
            print("Too little text, skipping.")
            continue                                     #Skip pages with too little text

        course_title = extract_course_title(title, course_code) #Extract clean course title
        structured = extract_course_structured_data(      #Extract useful metadata
            course_code=course_code,
            course_title=course_title,
            url=url,
            all_terms=all_terms,
            all_urls=all_urls,
            text=text,
        )

        chunks = split_into_chunks(text)                  #Split page into chunks

        for chunk_number, chunk in enumerate(chunks):
            chunk_id = f"{course_code}_{chunk_number}"    #Create unique chunk ID

            metadata = {                                  #Metadata stored with chunk
                **structured,
                "chunk_number": chunk_number,
            }

            #Text used for vector embedding
            embedding_text = f"""                         
                Course code: {course_code}
                Course title: {course_title}
                Terms found: {all_terms}
                Free-standing course: {structured["free_standing"]}
                Credits: {structured["credits"]}
                Assessment: {structured["assessment"]}

                Course content:
                {chunk}
                """

            embedding = get_embedding(embedding_text)     #Embed chunk text

            collection.upsert(                            #Store chunk in ChromaDB
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[metadata],
            )

            total_chunks += 1                             #Increase chunk counter

        indexed_pages += 1                                #Increase page counter

        print(
            f"Indexed {len(chunks)} chunks for {course_code}. "
            f"Free-standing: {structured['free_standing']}, "
            f"Credits: {structured['credits']}"
        )

        time.sleep(0.15)                                  #Small delay between pages

    print("\nFinished indexing.")
    print(f"Indexed unique IKT courses: {indexed_pages}")
    print(f"Indexed chunks: {total_chunks}")



#COURSE OVERVIEW HELPERS
def get_all_indexed_courses():
    collection = get_collection()                         #Open vector collection

    try:
        count = collection.count()                        #Count indexed items
    except Exception:
        return []                                         #Return empty on error

    if count == 0:
        return []                                         #Return empty if index is empty

    data = collection.get(include=["metadatas"])          #Load all metadata
    courses = {}                                          #Store unique courses

    for meta in data.get("metadatas", []):
        code = meta.get("course_code", "UNKNOWN")         #Get course code

        if code == "UNKNOWN":
            continue                                      #Skip invalid metadata

        if code not in courses:
            courses[code] = {                             #Store one entry per course
                "course_code": code,
                "title": meta.get("title", "No title"),
                "url": meta.get("url", "No URL"),
                "all_terms": meta.get("all_terms", "unknown"),
                "free_standing": meta.get("free_standing", "Unknown"),
                "credits": meta.get("credits", "Unknown"),
                "assessment": meta.get("assessment", "Unknown"),
                "learning_outcomes": meta.get("learning_outcomes", "Unknown"),
            }

    return [courses[code] for code in sorted(courses)]    #Return sorted course list


def format_course_list(courses):
    if not courses:
        return "No indexed IKT courses were found. Run: python main.py --reset --crawl"

    lines = [                                             #Start output list
        f"The indexed UiA dataset contains {len(courses)} unique IKT courses:\n"
    ]

    for course in courses:
        lines.append(                                     #Add one course entry
            f"- {course['course_code']} — {course['title']}\n"
            f"  Terms found: {course['all_terms']}\n"
            f"  Free-standing: {course['free_standing']}\n"
            f"  Credits: {course['credits']}\n"
            f"  Source: {course['url']}"
        )

    return "\n".join(lines)                               #Return formatted text


def format_structured_course_list(courses, heading):
    if not courses:
        return f"No matching courses were found for: {heading}"

    lines = [f"{heading} ({len(courses)} courses):\n"]    #Start output list

    for course in courses:
        lines.append(                                     #Add one course entry
            f"- {course['course_code']} — {course['title']}\n"
            f"  Free-standing: {course['free_standing']}\n"
            f"  Credits: {course['credits']}\n"
            f"  Source: {course['url']}"
        )

    return "\n".join(lines)                               #Return formatted text



#QUERY CLASSIFICATION
def expand_semantic_query(question):
    q = question.lower()

    if (
        "ai" in q
        or "artificial intelligence" in q
        or "kunstig intelligens" in q
        or "maskinlæring" in q
    ):
        return question + " artificial intelligence AI machine learning deep learning neural networks"

    if (
        "cybersecurity" in q
        or "security" in q
        or "cyber security" in q
        or "sikkerhet" in q
        or "datasikkerhet" in q
    ):
        return question + " security cybersecurity cryptography penetration testing risk vulnerability"

    if (
        "database" in q
        or "databases" in q
        or "databaser" in q
        or "data management" in q
    ):
        return question + " database data management big data"

    return question

def is_count_question(question):
    q = question.lower()                                  #Normalize question

    return (                                              #Detect counting questions
        "how many" in q
        or "count" in q
        or "number of" in q
        or "total" in q
    )


def is_list_question(question):
    q = question.lower()                                  #Normalize question

    return (                                              #Detect listing questions
        "list" in q
        or "show" in q
        or "all courses" in q
        or "all ikt courses" in q
        or "what ikt courses" in q
        or "which ikt courses" in q
    )


def is_general_course_catalog_question(question):
    q = question.lower()                                  #Normalize question

    has_course_word = (                                   #Check if user asks about courses
        "course" in q
        or "courses" in q
        or "subject" in q
        or "subjects" in q
    )

    has_catalog_word = (                                  #Check if query is about UiA/IKT catalog
        "ikt" in q
        or "uia" in q
        or "course" in q
        or "courses" in q
    )

    return has_course_word and has_catalog_word           #Return combined result


def is_free_standing_question(question):
    q = question.lower()                                  #Normalize question

    return (                                              #Detect free-standing questions
        "free-standing" in q
        or "freestanding" in q
        or "free standing" in q
    )


def is_assessment_question(question):
    q = question.lower()                                  #Normalize question

    return (                                              #Detect assessment questions
        "assessment" in q
        or "exam" in q
        or "examination" in q
        or "portfolio" in q
        or "oral" in q
        or "written" in q
    )


def is_credit_question(question):
    q = question.lower()                                  #Normalize question

    return (                                              #Detect credit questions
        "credit" in q
        or "credits" in q
        or "ects" in q
        or "study points" in q
    )


def classify_question(question):
    course_codes = extract_course_codes_from_question(question) #Extract course codes

    if len(course_codes) >= 2:
        return "MULTI_COURSE"                             #Comparison or multi-course query

    if len(course_codes) == 1:
        return "EXACT_COURSE"                             #Question about one course

    if is_free_standing_question(question):
        return "FREE_STANDING"                            #Structured free-standing query

    if is_count_question(question) and is_general_course_catalog_question(question):
        return "COUNT_COURSES"                            #Structured count query

    if is_list_question(question) and is_general_course_catalog_question(question):
        return "LIST_COURSES"                             #Structured list query

    return "SEMANTIC_SEARCH"                              #Default dense retrieval query



#DETERMINISTIC STRUCTURED ANSWERING
def answer_free_standing_question(question):
    courses = get_all_indexed_courses()                   #Load indexed courses

    if not courses:
        return "No indexed IKT courses were found. Run: python main.py --reset --crawl"

    q = question.lower()                                  #Normalize question

    wants_not_free_standing = (                           #Detect negative free-standing query
        "not free-standing" in q
        or "not freestanding" in q
        or "not free standing" in q
        or "non freestanding" in q
        or "non free standing" in q
        or "non-free-standing" in q
        or "non-free" in q
        or "are not free" in q
        or "is not free" in q
        or "aren't free" in q
    )

    if wants_not_free_standing:
        filtered = [                                      #Keep courses marked no
            course for course in courses
            if course["free_standing"].lower() == "no"
        ]
        return format_structured_course_list(
            filtered,
            "Courses marked as not free-standing",
        )

    filtered = [                                          #Keep courses marked yes
        course for course in courses
        if course["free_standing"].lower() == "yes"
    ]

    return format_structured_course_list(
        filtered,
        "Courses marked as free-standing",
    )


def answer_exact_course_structured(course_code, question):
    courses = get_all_indexed_courses()                   #Load indexed courses
    course = None                                         #Placeholder for match

    for candidate in courses:
        if candidate["course_code"] == course_code:
            course = candidate                            #Store matching course
            break

    if not course:
        return None                                       #No structured answer possible

    if is_free_standing_question(question):
        return (                                          #Return direct free-standing answer
            f"{course_code} — {course['title']}\n"
            f"Free-standing course: {course['free_standing']}\n"
            f"Source: {course['url']}"
        )

    if is_credit_question(question):
        return (                                          #Return direct credit answer
            f"{course_code} — {course['title']}\n"
            f"Credits: {course['credits']}\n"
            f"Source: {course['url']}"
        )

    if is_assessment_question(question):
        return (                                          #Return direct assessment answer
            f"{course_code} — {course['title']}\n"
            f"Assessment information:\n{course['assessment']}\n"
            f"Source: {course['url']}"
        )

    return None                                           #Use RAG if no structured rule matches



#RETRIEVAL
def make_retrieved_item(doc, meta, distance=0.0):
    return {                                              #Standard format for retrieved chunks
        "text": doc,
        "title": meta.get("title", "No title"),
        "url": meta.get("url", "No URL"),
        "all_urls": meta.get("all_urls", ""),
        "course_code": meta.get("course_code", "UNKNOWN"),
        "all_terms": meta.get("all_terms", "unknown"),
        "free_standing": meta.get("free_standing", "Unknown"),
        "credits": meta.get("credits", "Unknown"),
        "distance": distance,
    }


def retrieve_exact_course(course_code):
    collection = get_collection()                         #Open vector collection
    course_code = normalize_course_code(course_code)      #Normalize course code

    try:
        data = collection.get(                            #Fetch all chunks for this course
            where={"course_code": course_code},
            include=["documents", "metadatas"],
        )
    except Exception:
        return []                                         #Return empty on error

    documents = data.get("documents", [])                 #Get documents
    metadatas = data.get("metadatas", [])                 #Get metadata

    items = [                                             #Convert to retrieved item format
        make_retrieved_item(doc, meta, 0.0)
        for doc, meta in zip(documents, metadatas)
    ]

    return sorted(items, key=lambda item: item["course_code"]) #Return sorted chunks


def retrieve_multiple_courses(course_codes):
    retrieved = []                                        #Store chunks from all courses

    for course_code in course_codes:
        chunks = retrieve_exact_course(course_code)       #Retrieve chunks for one course

        if chunks:
            retrieved.extend(chunks[:MAX_CHUNKS_PER_COURSE]) #Use first chunks per course

    return retrieved                                      #Return combined chunks


def retrieve(question):
    collection = get_collection()                         #Open vector collection

    try:
        count = collection.count()                        #Count indexed chunks
    except Exception:
        count = 0                                         #Treat errors as empty collection

    if count == 0:
        return []                                         #Return empty if no index exists

    exact_code_match = COURSE_CODE_RE.search(question)    #Look for course code

    if exact_code_match:
        exact_code = normalize_course_code(exact_code_match.group(0)) #Normalize code
        exact_results = retrieve_exact_course(exact_code) #Retrieve exact course

        if exact_results:
            return exact_results                          #Return exact match if found

    expanded_question = expand_semantic_query(question)
    query_embedding = get_embedding(expanded_question)
    raw_n_results = min(max(TOP_K * 8, 50), count)        #Retrieve extra before filtering

    results = collection.query(                           #Dense vector search
        query_embeddings=[query_embedding],
        n_results=raw_n_results,
    )

    retrieved = []                                        #Store final retrieved chunks
    best_distance = None

    documents = results.get("documents", [[]])[0]         #Retrieved texts
    metadatas = results.get("metadatas", [[]])[0]         #Retrieved metadata
    distances = results.get("distances", [[]])[0]         #Retrieved distances
    
    if not distances or distances[0] > MAX_SEMANTIC_DISTANCE:
        return []

    course_chunk_counts = {}                              #Limit chunks per course

    for doc, meta, distance in zip(documents, metadatas, distances):
        if best_distance is None:
            best_distance = distance
        course_code = meta.get("course_code", "UNKNOWN")  #Get course code
        course_chunk_counts.setdefault(course_code, 0)    #Initialize count

        if course_chunk_counts[course_code] >= MAX_CHUNKS_PER_COURSE:
            continue                                      #Skip too many chunks from same course

        course_chunk_counts[course_code] += 1             #Increase course chunk count
        retrieved.append(make_retrieved_item(doc, meta, distance)) #Add result

        if len(retrieved) >= TOP_K:
            break                                         #Stop when enough chunks are found

        if best_distance is not None and best_distance > 0.50:
            return []

    return retrieved                                      #Return retrieved chunks



#GENERATION
def build_context(retrieved_chunks):
    parts = []                                            #Store source blocks

    for i, chunk in enumerate(retrieved_chunks, start=1):
        parts.append(                                     #Format one source for the LLM
            f"[Source {i}]\n"
            f"Course code: {chunk['course_code']}\n"
            f"Title: {chunk['title']}\n"
            f"Free-standing: {chunk['free_standing']}\n"
            f"Credits: {chunk['credits']}\n"
            f"URL: {chunk['url']}\n"
            f"Text: {chunk['text']}"
        )

    return "\n\n".join(parts)                            #Join all source blocks


def generate_answer(question, retrieved_chunks):
    context = build_context(retrieved_chunks)             #Build LLM context

    system_prompt = """
        You are a RAG chatbot for University of Agder IKT courses.

        Rules:
        - Always answer in English.
        - Use only the retrieved UiA course context.
        - Do not invent facts, course codes, semesters, course names, programming languages, or categories.
        - Do not infer or guess. If the context does not explicitly say something, answer that it is not explicitly stated in the retrieved context.
        - Use the exact course code prefix from the context. Do not translate IKT to ICT.
        - For comparison questions, write one short paragraph followed by bullet points with the most important academic differences. Do not use the headings "Similarities" or "Differences".
        - Compare academic content, focus, credits, free-standing status, learning outcomes, teaching methods, and assessment when the context supports it.
        - Avoid trivial comparisons such as both courses having the same prefix, same faculty, same duration, or both being courses.
        - If Norwegian text appears in the context, translate only the relevant information into English.
        - Mention relevant course codes.
        - Keep the answer concise.
        - Never infer or guess. If the retrieved context does not explicitly mention the answer, say: "The retrieved context does not explicitly state this."
        - For questions asking whether a course uses a specific tool, programming language, technology, or method, answer "yes" only if that exact item is explicitly mentioned in the retrieved context.
        """

    #Prompt with question and context
    user_prompt = f"""                                   
        Question:
        {question}

        Retrieved UiA context:
        {context}

        For comparison questions:
        Write the answer as:
        1. One short summary sentence.
        2. Then 3-5 bullet points with the most important academic contrasts.
        Do not use section headings.

        Answer in English:
        """

    response = ollama.chat(                               #Generate answer with LLM
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        options={"temperature": 0.0},                    #Use deterministic generation
    )

    return response["message"]["content"]                #Return generated answer


def ask_question(question):
    intent = classify_question(question)                  #Classify user question

    if intent == "MULTI_COURSE":
        course_codes = extract_course_codes_from_question(question) #Get course codes
        retrieved = retrieve_multiple_courses(course_codes) #Retrieve chunks for each course

        if not retrieved:
            return (
                "I could not find sufficiently relevant information in the indexed UiA IKT course data. "
                "The question may be outside the scope of this chatbot.",
                [],
            )

        answer = generate_answer(question, retrieved)     #Generate comparison answer
        return answer, retrieved

    if intent == "COUNT_COURSES":
        courses = get_all_indexed_courses()               #Load all courses
        return (
            f"The indexed UiA dataset contains {len(courses)} unique IKT courses.",
            [],
        )

    if intent == "LIST_COURSES":
        courses = get_all_indexed_courses()               #Load all courses
        return format_course_list(courses), []            #Return deterministic list

    if intent == "FREE_STANDING":
        return answer_free_standing_question(question), [] #Return deterministic answer

    if intent == "EXACT_COURSE":
        match = COURSE_CODE_RE.search(question)           #Find course code
        course_code = normalize_course_code(match.group(0)) #Normalize course code

        structured_answer = answer_exact_course_structured(course_code, question) #Try metadata answer

        if structured_answer:
            return structured_answer, []                  #Return direct answer

        retrieved = retrieve_exact_course(course_code)    #Retrieve course chunks

        if not retrieved:
            return f"No indexed information was found for {course_code}.", []

        answer = generate_answer(question, retrieved)     #Generate answer from course chunks
        return answer, retrieved

    retrieved = retrieve(question)                        #Use dense retrieval for general query

    if not retrieved:
        return (
            "I could not find sufficiently relevant information in the indexed UiA IKT course data. The question may be outside the scope of this chatbot.",
            [],
        )

    answer = generate_answer(question, retrieved)         #Generate answer from retrieved chunks
    return answer, retrieved



#INTERFACE
def print_retrieved(retrieved, max_to_print=8):
    print("\nRetrieved sources:")

    if not retrieved:
        print("No retrieval needed for this query.")
        return                                           #No sources to print

    seen_urls = set()                                    #Avoid printing duplicate URLs
    printed = 0                                          #Count printed sources

    for chunk in retrieved:
        url = chunk["url"]                               #Get source URL

        if url in seen_urls:
            continue                                     #Skip duplicate source

        seen_urls.add(url)                               #Mark source as printed
        printed += 1                                     #Increase printed count

        print(f"{printed}. {chunk['course_code']} | {chunk['title']}")
        print(f"   Free-standing: {chunk['free_standing']}")
        print(f"   Credits: {chunk['credits']}")
        print(f"   {chunk['url']}")
        print(f"   distance: {chunk['distance']:.4f}")

        if printed >= max_to_print:
            break                                        #Stop after max sources


def chat():
    print("\nUiA IKT RAG Chatbot")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Enter query: ").strip()        #Read user question

        if not question:
            print("Please enter a question.")            #Handle empty input
            continue

        if question.lower() in {"exit", "quit"}:
            print("Bye.")
            break                                        #Stop chat loop

        answer, retrieved = ask_question(question)        #Answer user question

        print_retrieved(retrieved)                       #Show retrieved sources
        print("\nAnswer:")
        print(answer)                                    #Show answer
        print()


def main():
    parser = argparse.ArgumentParser(description="UiA IKT RAG System") #Create CLI parser

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete old ChromaDB vector database.",
    )

    parser.add_argument(
        "--crawl",
        action="store_true",
        help="Discover, scrape, and index UiA IKT course pages.",
    )

    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start the chatbot.",
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help="Maximum unique IKT courses to index.",
    )

    args = parser.parse_args()                           #Read command-line arguments

    if args.reset:
        reset_database()                                 #Delete old database

    if args.crawl:
        crawl_and_index(args.max_pages)                  #Build new vector index

    if args.chat:
        chat()                                           #Start chat interface

    if not any([args.reset, args.crawl, args.chat]):
        print("Choose one option:")
        print("  python main.py --reset --crawl")
        print("  python main.py --chat")


if __name__ == "__main__":
    main()                                               #Run program
