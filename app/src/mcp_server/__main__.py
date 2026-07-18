import mcp_server.tools.arasaac
import mcp_server.tools.time_tool
import mcp_server.tools.schedule_tool
from mcp_server.server import mcp

if __name__ == "__main__":
    mcp.run(transport="stdio")
