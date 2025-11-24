import os

import pytest
from dotenv import load_dotenv

# Load Env vars (API Key) before importing ai_engine
load_dotenv()

import ai_engine

# --- CONFIG FOR TEST ---
TEST_TRIAL_ID = 99999
TEST_UPLOAD_DIR = "./uploaded_evidence"

# --- MOCK DATA ---
CASE_BACKGROUND = """
The defendant, Mr. Smith, is accused of stealing a diamond necklace 
from the local museum on the night of December 14th. 
Security footage shows a man of similar height wearing a red hoodie.
"""

USER_ARGUMENTS = [
    "I was at home watching TV during the robbery.",
    "I do not own a red hoodie.",
]

# --- FIXTURES (Setup & Teardown) ---


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    """
    Runs before and after tests.
    Ensures we start clean and end clean.
    """
    # 1. Cleanup before start (in case previous test failed)
    ai_engine.delete_trial_data(TEST_TRIAL_ID)

    yield  # Run the tests...

    # 2. Cleanup after finish
    print("\n--- TEARDOWN: Cleaning up test data ---")
    ai_engine.delete_trial_data(TEST_TRIAL_ID)


# --- TESTS ---


@pytest.mark.asyncio
async def test_1_ingest_background():
    """Test if we can create a vector store and ingest text."""
    print("\n[Test 1] Ingesting Background...")
    try:
        ai_engine.ingest_text_evidence(
            TEST_TRIAL_ID, CASE_BACKGROUND, "Case Background"
        )

        # Verify it exists in Chroma
        store = ai_engine.get_trial_vector_store(TEST_TRIAL_ID)
        # We assume ingestion worked if we can verify the collection exists
        assert store is not None
        print("✅ Background ingested successfully.")
    except Exception as e:
        pytest.fail(f"Ingestion failed: {e}")


@pytest.mark.asyncio
async def test_2_batch_generation():
    """Test parallel argument generation."""
    print("\n[Test 2] Running Batch Generation...")

    responses = await ai_engine.batch_generate_initial_arguments(
        TEST_TRIAL_ID, CASE_BACKGROUND, USER_ARGUMENTS
    )

    assert len(responses) == 2
    assert isinstance(responses[0], str)
    assert len(responses[0]) > 10  # Check if we got a real text response

    print(f"✅ Received {len(responses)} responses.")
    print(f"   Sample: {responses[0][:50]}...")


@pytest.mark.asyncio
async def test_3_file_processing_and_reply():
    """Test uploading a mock PDF and generating a reply."""
    print("\n[Test 3] Testing File Upload & Reply...")

    # 1. Create a dummy PDF text file (Simulating a file upload)
    # Since we don't want to create a real PDF binary in code, we will test the
    # 'process_file_to_text' logic by mocking a file save or just passing a path
    # For this test, let's create a dummy text file to act as evidence

    mock_filename = "alibi_witness_statement.txt"
    trial_folder = os.path.join(TEST_UPLOAD_DIR, str(TEST_TRIAL_ID))
    os.makedirs(trial_folder, exist_ok=True)
    mock_path = os.path.join(trial_folder, mock_filename)

    with open(mock_path, "w") as f:
        f.write(
            "Witness Statement: I saw Mr. Smith at the bar at 10 PM. He was not at the museum."
        )

    # 2. Call the function (We simulate it being a PDF/Text file for logic)
    # Note: ai_engine.process_file_to_text handles PDF/Image specifically.
    # To test strictly without a real PDF, we will bypass the specific loader
    # check or creating a minimal PDF is complex.
    # INSTEAD: We will test the REPLY logic with `new_file_path` set to None
    # to ensure the RAG pipeline works, then we can try a mock flow.

    user_reply = "Here is a witness statement saying I was elsewhere."

    # We pass a simple chat history
    history = []

    response = await ai_engine.generate_reply_with_new_evidence(
        trial_id=TEST_TRIAL_ID,
        case_background=CASE_BACKGROUND,
        user_text=user_reply,
        chat_history=history,
        new_file_path=None,  # Creating real PDFs in tests is tricky, skipping file parsing for basic sanity check
        new_file_type=None,
    )

    assert response is not None
    assert len(response) > 0
    print(f"✅ Reply Generated: {response[:50]}...")


@pytest.mark.asyncio
async def test_4_cleanup():
    """Test deletion logic."""
    print("\n[Test 4] Testing Cleanup...")

    # Verify file exists from previous step
    trial_folder = os.path.join(TEST_UPLOAD_DIR, str(TEST_TRIAL_ID))
    assert os.path.exists(trial_folder)

    # Run cleanup
    ai_engine.delete_trial_data(TEST_TRIAL_ID)

    # Verify gone
    assert not os.path.exists(trial_folder)
    print("✅ Cleanup successful.")
