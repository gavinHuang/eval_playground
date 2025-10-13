"""
Snowflake Cortex Analyst REST API Client

This script provides a client for interacting with Snowflake's Cortex Analyst REST API.
Reference: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst/rest-api

Dependencies:
- requests
- python-dotenv (optional, for loading .env files)

Setup:
1. Install dependencies: pip install requests python-dotenv
2. Copy .env.example to .env and configure your Snowflake credentials
3. Run: python snowflake_cortex_analyst.py
"""

import json
import os
import requests
from typing import Dict, List, Optional, Union, Generator, Any
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, skip loading .env file
    pass

try:
    import jwt
    import time
    import hashlib
    import base64
    from cryptography.hazmat.primitives import serialization
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False


class ContentType(Enum):
    """Content types supported in messages"""
    TEXT = "text"
    SQL = "sql"
    SUGGESTIONS = "suggestions"


class Role(Enum):
    """Message roles"""
    USER = "user"
    ANALYST = "analyst"


@dataclass
class MessageContent:
    """Content object for a message"""
    type: str
    text: Optional[str] = None
    statement: Optional[str] = None
    suggestions: Optional[List[str]] = None
    confidence: Optional[Dict] = None

    def to_dict(self):
        """Convert to dictionary, excluding None values"""
        result = {"type": self.type}
        if self.text is not None:
            result["text"] = self.text
        if self.statement is not None:
            result["statement"] = self.statement
        if self.suggestions is not None:
            result["suggestions"] = self.suggestions
        if self.confidence is not None:
            result["confidence"] = self.confidence
        return result


@dataclass
class Message:
    """Message object"""
    role: str
    content: List[MessageContent]

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "role": self.role,
            "content": [c.to_dict() for c in self.content]
        }


class CortexAnalystClient:
    """Client for Snowflake Cortex Analyst REST API"""

    def __init__(
        self,
        account_url: str,
        token: str,
        token_type: str = "OAUTH"
    ):
        """
        Initialize the Cortex Analyst client.

        Args:
            account_url: Snowflake account URL (e.g., https://your-account.snowflakecomputing.com)
            token: Authorization token (OAuth token, JWT token, or PAT)
            token_type: Token type - "OAUTH", "KEYPAIR_JWT", or "PROGRAMMATIC_ACCESS_TOKEN"
        """
        self.account_url = account_url.rstrip('/')
        self.token = token
        self.token_type = token_type
        self.base_url = f"{self.account_url}/api/v2/cortex/analyst"

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Snowflake-Authorization-Token-Type": self.token_type
        }
        return headers

    def send_message(
        self,
        question: str,
        semantic_model_file: Optional[str] = None,
        semantic_model: Optional[str] = None,
        semantic_models: Optional[List[Dict[str, str]]] = None,
        semantic_view: Optional[str] = None,
        conversation_history: Optional[List[Message]] = None,
        stream: bool = False
    ) -> Union[Dict, Generator]:
        """
        Send a message to Cortex Analyst.

        Args:
            question: The user's question
            semantic_model_file: Path to semantic model YAML file on stage
                                (e.g., @my_db.my_schema.my_stage/my_semantic_model.yaml)
            semantic_model: YAML string containing the semantic model
            semantic_models: List of semantic models/views to choose from
                           (e.g., [{"semantic_view": "DB.SCH.VIEW"}])
            semantic_view: Fully qualified name of semantic view
            conversation_history: Previous messages in the conversation
            stream: Whether to stream the response

        Returns:
            Dict with response (non-streaming) or Generator of events (streaming)
        """
        # Build request body
        messages = []
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                messages.append(msg.to_dict())
        
        # Add current question
        messages.append({
            "role": Role.USER.value,
            "content": [
                {
                    "type": ContentType.TEXT.value,
                    "text": question
                }
            ]
        })

        request_body = {"messages": messages}

        # Add semantic model specification (must specify exactly one)
        if semantic_model_file:
            request_body["semantic_model_file"] = semantic_model_file
        elif semantic_model:
            request_body["semantic_model"] = semantic_model
        elif semantic_models:
            request_body["semantic_models"] = semantic_models
        elif semantic_view:
            request_body["semantic_view"] = semantic_view
        else:
            raise ValueError(
                "Must specify one of: semantic_model_file, semantic_model, "
                "semantic_models, or semantic_view"
            )

        if stream:
            request_body["stream"] = True

        # Send request
        url = f"{self.base_url}/message"
        headers = self._get_headers()

        if stream:
            return self._stream_response(url, headers, request_body)
        else:
            try:
                response = requests.post(url, headers=headers, json=request_body)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                # Print the raw response for debugging
                print(f"HTTP {response.status_code} Error:")
                try:
                    error_response = response.json()
                    print(json.dumps(error_response, indent=2))
                except:
                    print(response.text)
                raise

    def _stream_response(
        self,
        url: str,
        headers: Dict[str, str],
        request_body: Dict
    ) -> Generator[Dict, None, None]:
        """
        Stream response using server-sent events.

        Args:
            url: API endpoint URL
            headers: Request headers
            request_body: Request body

        Yields:
            Parsed event dictionaries
        """
        response = requests.post(
            url,
            headers=headers,
            json=request_body,
            stream=True
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('event:'):
                    event_type = line[6:].strip()
                elif line.startswith('data:'):
                    data = line[5:].strip()
                    try:
                        data_json = json.loads(data)
                        yield {
                            "event": event_type,
                            "data": data_json
                        }
                    except json.JSONDecodeError:
                        # Skip malformed JSON
                        continue

    def send_feedback(
        self,
        request_id: str,
        positive: bool,
        feedback_message: Optional[str] = None
    ) -> bool:
        """
        Send feedback about a response.

        Args:
            request_id: The request_id from the message response
            positive: True for positive feedback, False for negative
            feedback_message: Optional feedback message

        Returns:
            True if successful
        """
        url = f"{self.base_url}/feedback"
        headers = self._get_headers()

        request_body = {
            "request_id": request_id,
            "positive": positive
        }

        if feedback_message:
            request_body["feedback_message"] = feedback_message

        response = requests.post(url, headers=headers, json=request_body)
        response.raise_for_status()
        return response.status_code == 200

    @staticmethod
    def parse_response(response: Dict) -> Dict:
        """
        Parse a non-streaming response into structured data.

        Args:
            response: Response dictionary from send_message

        Returns:
            Parsed response with extracted text, SQL, suggestions, and query results
        """
        result = {
            "request_id": response.get("request_id"),
            "text": [],
            "sql": None,
            "suggestions": [],
            "warnings": response.get("warnings", []),
            "response_metadata": response.get("response_metadata", {}),
            "query_results": None
        }

        message = response.get("message", {})
        content = message.get("content", [])

        for item in content:
            content_type = item.get("type")
            if content_type == ContentType.TEXT.value:
                result["text"].append(item.get("text", ""))
            elif content_type == ContentType.SQL.value:
                result["sql"] = {
                    "statement": item.get("statement", ""),
                    "confidence": item.get("confidence")
                }
            elif content_type == ContentType.SUGGESTIONS.value:
                result["suggestions"] = item.get("suggestions", [])
        
        # Extract query results from response_metadata if available
        response_metadata = response.get("response_metadata", {})
        if "query_result" in response_metadata:
            query_result = response_metadata["query_result"]
            # Convert to list of tuples format for consistency with SQLite results
            if "rows" in query_result:
                result["query_results"] = [
                    tuple(row.values()) if isinstance(row, dict) else tuple(row)
                    for row in query_result["rows"]
                ]

        return result


def generate_jwt_token(
    account_identifier: str,
    username: str,
    private_key_path: str,
    private_key_passphrase: Optional[str] = None
) -> str:
    """
    Generate a JWT token for key-pair authentication.
    
    Args:
        account_identifier: Snowflake account identifier (e.g., "MYORG-MYACCOUNT")
        username: Snowflake username (will be converted to uppercase)
        private_key_path: Path to the private key file (.p8 format)
        private_key_passphrase: Passphrase for encrypted private key (if any)
    
    Returns:
        JWT token string
        
    Raises:
        ImportError: If PyJWT and cryptography packages are not installed
        ValueError: If the private key cannot be loaded
    """
    if not JWT_AVAILABLE:
        raise ImportError(
            "JWT token generation requires PyJWT and cryptography packages. "
            "Install with: pip install PyJWT[crypto] cryptography"
        )
    
    # Read and parse the private key
    try:
        with open(private_key_path, 'rb') as key_file:
            private_key_data = key_file.read()
        
        if private_key_passphrase:
            private_key_passphrase = private_key_passphrase.encode()
        
        private_key = serialization.load_pem_private_key(
            private_key_data,
            password=private_key_passphrase
        )
        
        # Get the public key and calculate its fingerprint
        public_key = private_key.public_key()
        public_key_der = public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Calculate SHA256 fingerprint
        sha256_hash = hashlib.sha256(public_key_der).digest()
        public_key_fp = base64.b64encode(sha256_hash).decode('utf-8')
        
    except Exception as e:
        raise ValueError(f"Failed to load private key from {private_key_path}: {e}")
    
    # Prepare JWT payload
    account_identifier = account_identifier.upper()
    username = username.upper()
    
    now = int(time.time())
    payload = {
        'iss': f"{account_identifier}.{username}.SHA256:{public_key_fp}",
        'sub': f"{account_identifier}.{username}",
        'iat': now,
        'exp': now + 3600  # Token expires in 1 hour
    }
    
    # Generate the JWT token
    token = jwt.encode(payload, private_key, algorithm='RS256')
    return token


def create_client_from_keypair(
    account_url: str,
    account_identifier: str,
    username: str,
    private_key_path: str,
    private_key_passphrase: Optional[str] = None
) -> 'CortexAnalystClient':
    """
    Create a CortexAnalystClient using key-pair authentication.
    
    Args:
        account_url: Snowflake account URL
        account_identifier: Account identifier for JWT (e.g., "MYORG-MYACCOUNT")
        username: Snowflake username
        private_key_path: Path to private key file
        private_key_passphrase: Passphrase for encrypted private key
    
    Returns:
        Configured CortexAnalystClient instance
    """
    jwt_token = generate_jwt_token(
        account_identifier=account_identifier,
        username=username,
        private_key_path=private_key_path,
        private_key_passphrase=private_key_passphrase
    )
    
    return CortexAnalystClient(
        account_url=account_url,
        token=jwt_token,
        token_type="KEYPAIR_JWT"
    )


class CortexAnalystWrapper:
    """Wrapper for Cortex Analyst with unified interface for evaluation"""
    
    def __init__(
        self,
        account_url: str,
        token: str,
        semantic_model_file: str,
        token_type: str = "OAUTH",
        warehouse: Optional[str] = None,
        database: Optional[str] = None,
        schema: Optional[str] = None
    ):
        """
        Initialize wrapper
        
        Args:
            account_url: Snowflake account URL
            token: Authorization token
            semantic_model_file: Path to semantic model file on stage
            token_type: Token type
            warehouse: Default warehouse for SQL execution (optional)
            database: Default database for SQL execution (optional)
            schema: Default schema for SQL execution (optional)
        """
        self.client = CortexAnalystClient(
            account_url=account_url,
            token=token,
            token_type=token_type
        )
        self.account_url = account_url
        self.token = token
        self.token_type = token_type
        self.semantic_model_file = semantic_model_file
        self.warehouse = warehouse
        self.database = database
        self.schema = schema
    
    def query(self, question: str, evidence: Optional[str] = None) -> Dict[str, Any]:
        """
        Query interface compatible with evaluator
        
        Args:
            question: Natural language question
            evidence: Additional context to include with the question
            
        Returns:
            Dictionary with raw response including message.content for SQL extraction
        """
        try:
            # Combine question with evidence if provided
            full_question = question
            if evidence:
                full_question = f"{question}\n\nAdditional context: {evidence}"
            
            response = self.client.send_message(
                question=full_question,
                semantic_model_file=self.semantic_model_file
            )
            
            # Return the raw response with message.content intact
            # The evaluator will extract SQL from message.content
            return {
                "message": response.get("message", {}),
                "request_id": response.get("request_id"),
                "warnings": response.get("warnings", []),
                "response_metadata": response.get("response_metadata", {}),
                "error": None
            }
        except Exception as e:
            return {
                "message": {},
                "request_id": None,
                "warnings": [],
                "response_metadata": {},
                "error": str(e)
            }
    
    def execute_snowflake_sql(
        self, 
        sql: str,
        warehouse: Optional[str] = None,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        timeout: int = 60
    ) -> tuple[Optional[List[tuple]], Optional[str]]:
        """
        Execute SQL query against Snowflake using SQL REST API
        
        Reference: https://docs.snowflake.com/en/developer-guide/sql-api/submitting-requests
        
        Args:
            sql: SQL query to execute
            warehouse: Warehouse to use (uses instance default if not specified)
            database: Database to use (uses instance default if not specified)
            schema: Schema to use (uses instance default if not specified)
            timeout: Request timeout in seconds
            
        Returns:
            Tuple of (results, error). Results is a list of tuples or None if error.
        """
        try:
            # Build request URL
            url = f"{self.account_url}/api/v2/statements"
            
            # Build request headers
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Snowflake-Authorization-Token-Type": self.token_type
            }
            
            # Build request body
            request_body = {
                "statement": sql,
                "timeout": timeout
            }
            
            # Add warehouse (use parameter or instance default)
            wh = warehouse or self.warehouse
            if wh:
                request_body["warehouse"] = wh
            
            # Add database (use parameter or instance default)
            db = database or self.database
            if db:
                request_body["database"] = db
            
            # Add schema (use parameter or instance default)
            sch = schema or self.schema
            if sch:
                request_body["schema"] = sch
            
            # Submit SQL statement
            response = requests.post(url, headers=headers, json=request_body, timeout=timeout)
            response.raise_for_status()
            
            result_data = response.json()
            
            # # Check if the statement succeeded
            # if result_data.get("statementStatusUrl"):
            #     # Statement is still running, poll for results
            #     status_url = f"{self.account_url}{result_data['statementStatusUrl']}"
                
            #     import time
            #     max_polls = 30
            #     poll_interval = 2
                
            #     for _ in range(max_polls):
            #         time.sleep(poll_interval)
            #         status_response = requests.get(status_url, headers=headers, timeout=timeout)
            #         status_response.raise_for_status()
            #         result_data = status_response.json()
                    
            #         if result_data.get("statementStatus") in ["success", "failed"]:
            #             break
            
            # Check final status - sqlState "00000" means success
            sql_state = result_data.get("sqlState", "")
            if sql_state != "00000":
                error_msg = result_data.get("message", f"SQL execution failed with sqlState: {sql_state}")
                return None, error_msg
            
            # Extract results from response
            results = []
            
            # Check if data is directly in response (synchronous execution)
            if "data" in result_data:
                for row_data in result_data["data"]:
                    # Convert row data to tuple
                    row_tuple = tuple(row_data)
                    results.append(row_tuple)
            # Otherwise check for resultSet (may be from polling)
            elif "resultSet" in result_data:
                result_set = result_data["resultSet"]
                
                # Extract rows - each row is a list of values
                if "data" in result_set:
                    for row_data in result_set["data"]:
                        # Convert row data to tuple
                        row_tuple = tuple(row_data)
                        results.append(row_tuple)
            
            return results, None
            
        except requests.exceptions.Timeout:
            return None, f"Request timeout after {timeout} seconds"
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP {e.response.status_code} Error"
            try:
                error_response = e.response.json()
                if "message" in error_response:
                    error_msg += f": {error_response['message']}"
            except:
                error_msg += f": {e.response.text}"
            return None, error_msg
        except Exception as e:
            return None, str(e)


def main():
    """Example usage"""
    # Configuration from environment variables or .env file
    # Create a .env file with:
    # SNOWFLAKE_ACCOUNT_URL=https://your-account.snowflakecomputing.com
    # SNOWFLAKE_TOKEN=your-oauth-token-or-jwt-or-pat
    # SNOWFLAKE_TOKEN_TYPE=OAUTH  # or KEYPAIR_JWT or PROGRAMMATIC_ACCESS_TOKEN
    account_url = os.environ.get("SNOWFLAKE_ACCOUNT_URL")
    token = os.environ.get("SNOWFLAKE_TOKEN")
    token_type = os.environ.get("SNOWFLAKE_TOKEN_TYPE", "OAUTH")
    
    if not account_url or not token:
        print("Error: SNOWFLAKE_ACCOUNT_URL and SNOWFLAKE_TOKEN environment variables must be set")
        return

    # Check if using key-pair authentication with separate credentials
    account_identifier = os.environ.get("SNOWFLAKE_ACCOUNT_IDENTIFIER")
    username = os.environ.get("SNOWFLAKE_USERNAME")
    private_key_path = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH")
    private_key_passphrase = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
    
    if account_identifier and username and private_key_path:
        print("Using key-pair authentication with private key...")
        try:
            client = create_client_from_keypair(
                account_url=account_url,
                account_identifier=account_identifier,
                username=username,
                private_key_path=private_key_path,
                private_key_passphrase=private_key_passphrase
            )
            print(f"Generated JWT token for {username}@{account_identifier}")
        except Exception as e:
            print(f"Failed to create client with key-pair authentication: {e}")
            return
    else:
        # Use provided token
        client = CortexAnalystClient(account_url=account_url, token=token, token_type=token_type)

    # Example 1: Query with semantic model file
    print("Testing Cortex Analyst API...")
    print("-" * 40)
    try:
        response = client.send_message(
            question="How many gas stations in CZE has Premium gas?",
            semantic_model_file="@DEBIT_CARD_SPECIALIZING.public.semantic_yaml/debit_card_semantic_model.yaml"
            # semantic_model="debit_card_semantic_model.yaml"
        )
        
        # Print the raw response
        print("Response:")
        print(json.dumps(response, indent=2))

        # Extract SQL from response message.content
        sql_statement = None
        if "message" in response and "content" in response["message"]:
            for content_item in response["message"]["content"]:
                if content_item.get("type") == "sql":
                    sql_statement = content_item.get("statement")
                    break
        
        if sql_statement:
            print(f"\n{'-'*40}")
            print("Extracted SQL:")
            print(sql_statement)
            
            # Execute SQL against Snowflake using REST API
            warehouse = os.environ.get("SNOWFLAKE_WAREHOUSE")
            database = os.environ.get("SNOWFLAKE_DATABASE", "DEBIT_CARD_SPECIALIZING")
            schema_name = os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC")
            
            if not warehouse:
                print("\nError: SNOWFLAKE_WAREHOUSE environment variable must be set")
            else:
                # Create a wrapper instance with SQL execution configuration
                wrapper = CortexAnalystWrapper(
                    account_url=account_url,
                    token=token,
                    semantic_model_file="@DEBIT_CARD_SPECIALIZING.public.semantic_yaml/debit_card_semantic_model.yaml",
                    token_type=token_type,
                    warehouse=warehouse,
                    database=database,
                    schema=schema_name
                )
                
                print(f"\nExecuting SQL against Snowflake using REST API...")
                results, error = wrapper.execute_snowflake_sql(sql_statement)
                
                if error:
                    print(f"\nError executing SQL: {error}")
                else:
                    print(f"\n{'-'*40}")
                    print(f"Query Results ({len(results)} rows):")
                    for row in results:
                        print(row)
        else:
            print("\nNo SQL statement found in response")
        
    except Exception as e:
        print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
