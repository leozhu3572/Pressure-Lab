# Pressure Lab API Documentation

## 1. Trial Management

### Create a New Trial
* **Method:** `POST`
* **Endpoint:** `/trials/`
* **Content-Type:** `application/json`
* **Body:**
    ```json
    {
      "title": "The Cookie Theft",
      "case_background": "The jar was stolen from the kitchen...",
      "initial_arguments": ["I was watching TV", "I am allergic"] 
    }
    ```
* **Return:**
    ```json
    {
      "trial_id": 1,
      "status": "created",
      "threads_generated": 2
    }
    ```

### Get All Trials
* **Method:** `GET`
* **Endpoint:** `/trials/`
* **Return:** 
    ```json
    {
      "id": 1,
      "title": "The Cookie Theft",
      "case_background": "...",
      "threads": [
        { 
          "id": 10, 
          "title": "I was watching TV", 
          "messages": [
            {
              "sender": "user",
              "content": "I was watching TV"
            },
            {
              "sender": "ai",
              "content": "However, the TV logs show it was off...",
              "sources": ["TV_Logs.pdf", "Case Background"] 
            }
          ] 
        }
      ]
    }
    ```

### Get Trial Dashboard (Full Details)
* **Method:** `GET`
* **Endpoint:** `/trials/{trial_id}`
* **Return:**
    ```json
    {
      "id": 1,
      "title": "...",
      "case_background": "...",
      "threads": [
        { "id": 10, "title": "I was watching TV", "messages": [...] }
      ]
    }
    ```

### Update Case Info (Re-index AI)
* **Method:** `PUT`
* **Endpoint:** `/trials/{trial_id}`
* **Body:** `{ "case_background": "Updated text..." }`
* **Return:** `{ "status": "updated" }`

### Delete Trial
* **Method:** `DELETE`
* **Endpoint:** `/trials/{trial_id}`
* **Return:** `{ "status": "deleted" }`

---

## 2. Discussion Loop

### Reply to Thread (with Optional File)
* **Method:** `POST`
* **Endpoint:** `/trials/{trial_id}/threads/{thread_id}/reply`
* **Content-Type:** `multipart/form-data`
* **Body (Form Data):**
    * `content`: (Text) "I disagree because..."
    * `file`: (File Object) *Optional PDF or Image*
    * `history_limit`: (Int) Number of past messages to remember (Default: 4).
* **Return:**
    ```json
  {
    "user_content": "I disagree... [Attached: proof.pdf]",
    "ai_response": "However, looking at the new evidence...",
    "sources": [
      "proof.pdf", 
      "previous_contract.docx", 
      "Case Background"
    ]
  }
    ```

### Delete Thread
* **Method:** `DELETE`
* **Endpoint:** `/threads/{thread_id}`
* **Return:** `{ "status": "thread deleted" }`

---

## 3. Editing

### Edit Message (Prunes Future & Regenerates)
* **Method:** `PUT`
* **Endpoint:** `/messages/{message_id}`
* **Body:** `{ "new_content": "Corrected argument text", "history_limit": 4 }`
* **Return:**
    ```json
  {
    "status": "edited and regenerated",
    "ai_response": "New counter-argument based on edit...",
    "sources": ["Case Background", "witness_statement.txt"]
  }
    ```

### Delete Evidence File
* **Method:** `DELETE`
* **Endpoint:** `/trials/{trial_id}/files/{filename}`
* **Return:** `{ "status": "file removed" }`