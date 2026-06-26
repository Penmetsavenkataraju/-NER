OUTPUT UI 


<img width="1920" height="1020" alt="Screenshot 2026-06-26 151303" src="https://github.com/user-attachments/assets/30238e12-c6c7-41ed-9fa5-441a69959579" />
<img width="1920" height="1020" alt="Screenshot 2026-06-26 151321" src="https://github.com/user-attachments/assets/0f822f9c-3e61-4e8a-949b-f9a5e5c89540" />
<img width="1920" height="1020" alt="Screenshot 2026-06-26 151340" src="https://github.com/user-attachments/assets/b58d383d-4fe3-4faa-87d0-60a6ce104e48" />
<img width="1920" height="1020" alt="Screenshot 2026-06-26 151345" src="https://github.com/user-attachments/assets/d288b3ff-f38a-4e82-b5a6-4d4356028386" />
<img width="1920" height="1020" alt="Screenshot 2026-06-26 151400" src="https://github.com/user-attachments/assets/404ef7e6-c65b-446d-9c55-34854896dab9" />







PROJECT STEP BY STEP EXPLAINATION:=

1. Start with a sentence
   The user enters a sentence in the web app.
2. Preprocess the text
   The app converts the text to lowercase.
   It replaces some multi-word phrases with a single format for easier matching.
3. Split the sentence into words
   The app tokenizes the sentence into individual words.
4. Remove common words
   Words like “the”, “and”, “have” are removed because they are not important for medical term detection.
5. Match words with clinical terms
   Each remaining word is compared with a list of clinical terms.
   If the match is strong enough, it is treated as a clinical term.
6. Translate the detected terms
   The identified medical terms are translated into Spanish and Catalan.
7. Display the results
   The app shows the detected terms and their translations in the browser.


“This project is a Streamlit-based NLP application for clinical text analysis. It takes an English sentence, identifies medical terms using a predefined dictionary and fuzzy matching, and translates them into Spanish and Catalan using translation models. The results are displayed in a web interface so users can easily see the detected terms and their translations.”

