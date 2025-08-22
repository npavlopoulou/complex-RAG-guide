import os
from typing import Any

import dotenv


class ConfigParser:
    """This is a class for parsing the config file."""

    def __init__(
        self,
        databricks_host: str = None,
        databricks_token: str = None,
        databricks_claude_endpoint: str = None,
    ):
        """The constructor for ConfigParser class.

        Parameters:
            databricks_host (string) -- the host of Databricks (e.g., https://<your-databricks-domain>.cloud.databricks.com).
            databricks_token (string) -- the token of Databricks (e.g., dapixxxxxxxxxxxxxxxxxxxxxxxxxx).
            databricks_claude_endpoint (string) -- the endpoint of Claude LLM on Databricks (e.g., <your-endpoint-name>).
        """

        self.databricks_host = databricks_host
        self.databricks_token = databricks_token
        self.databricks_claude_endpoint = databricks_claude_endpoint
        dotenv.load_dotenv()  # parse .env file

    def parse_config_file(self) -> Any:
        """The function to parse the config file.

        Returns:
            config (any) -- the configuration parameters.
        """

        config = {
            "DATABRICKS": {
                "HOST": "",
                "TOKEN": "",
                "CLAUDE_ENDPOINT": ""
            }
        }

        # set up the correct environment variables
        if (
            self.databricks_host is None
            and self.databricks_token is None
            and self.databricks_claude_endpoint is None
        ):
            config["DATABRICKS"]["HOST"] = os.environ.get(
                "DATABRICKS_HOST", config["DATABRICKS"]["HOST"]
            )
            config["DATABRICKS"]["TOKEN"] = os.environ.get(
                "DATABRICKS_TOKEN", config["DATABRICKS"]["TOKEN"]
            )
            config["DATABRICKS"]["CLAUDE_ENDPOINT"] = os.environ.get(
                "CLAUDE_LLM_ENDPOINT_NAME", config["DATABRICKS"]["CLAUDE_ENDPOINT"]
            )
        else:
            config["DATABRICKS"]["HOST"] = self.databricks_host
            config["DATABRICKS"]["TOKEN"] = self.databricks_token
            config["DATABRICKS"]["CLAUDE_ENDPOINT"] = self.databricks_claude_endpoint

        if not isinstance(
                config["DATABRICKS"]["HOST"],
                str,
            ) or not isinstance(
                config["DATABRICKS"]["TOKEN"],
                str,
            ) or not isinstance(
                config["DATABRICKS"]["CLAUDE_ENDPOINT"],
                str,
            ):
            raise ValueError("Databricks host, token, and LLM endpoints must be strings.")

        return config