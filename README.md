# PakCato
A package category identification tool supporting multi-categorization.

## Approach
PakCato performs top-down semantic distance scoring using predefined categories and their keywords against metadata queries, such as GitHub README/topics, or PyPI summary/description/topics. Once the ranking is calculated by cosine similarity, the final category is determined by hybrid OPTICS and standard deviation-based clustering.


## Quick Start

1. **Install dependencies**  
   ```bash
   uv pip install -r uv.lock
   ```

2. **Set up API keys**  
   Secrets are managed using `python-dotenv`.  
   Create a `.env` file with your `OPENAI_API_KEY` (required) and `GITHUB_API_KEY` (optional).

3. **Run the demo**  
   Open `demo.ipynb` and follow the cells to see semantic categorization in action.


## Configuration

- **Categories and keywords:**  
  Edit `settings/category_keywords.toml` to define or adjust the semantic categories.

- **Embeddings:**  
  Currently uses OpenAI `text-embedding-3-small`. Other embeddings are supported by modifying `src/utils.py` (specifically the `get_emb_batch` function).


## Output

**Schema:**
```json
{
  "type": "object",
  "properties": {
    "winner": {
      "type": "array",
      "items": { "type": "string" },
      "description": "List of category names with the best clustered context distance scores."
    },
    "prediction_cluster_raw": {
      "type": "object",
      "patternProperties": {
        "^[0-9]+$": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Cluster ID."
        }
      },
      "description": "Mapping of cluster index (as string) to list of category names in that cluster."
    },
    "prediction_distance_raw": {
      "type": "object",
      "additionalProperties": {
        "type": "number"
      },
      "description": "Distance score between query and category name."
    }
  },
  "required": ["winner", "prediction_cluster_raw", "prediction_distance_raw"]
}
```

**Example:**
```json
{
  "winner": [
    "Machine Learning & AI Frameworks"
  ],
  "prediction_cluster_raw": {
    "1": [
      "Machine Learning & AI Frameworks"
    ],
    "2": [
      "Testing & Quality Validation"
    ],
    "3": [
      "Logging & Monitoring"
    ],
    "4": [
      "Configuration Management",
      "Database Interaction",
      "Operating System Interaction",
      "Authentication & Authorization"
    ],
    "5": [
      "File & Data Handling",
      "Web Framework Components"
    ],
    "6": [
      "Network Communication",
      "Cryptography",
      "Memory Management"
    ]
  },
  "prediction_distance_raw": {
    "Machine Learning & AI Frameworks": 0.2428405722771753,
    "Testing & Quality Validation": 0.16932297212452993,
    "Logging & Monitoring": 0.12828668472159763,
    "Configuration Management": 0.10654633356608996,
    "Database Interaction": 0.09074698306559636,
    "Operating System Interaction": 0.08958592405704817,
    "Authentication & Authorization": 0.08374149075198833,
    "File & Data Handling": 0.07484601150332962,
    "Web Framework Components": 0.07280569912243454,
    "Network Communication": 0.06941232214702157,
    "Cryptography": 0.06896012325134647,
    "Memory Management": 0.056110720002535856
  }
}
```
