# [Gemini CLI](https://geminicli.com/)

```bash
# 安装
$ npm install -g @google/gemini-cli

$ cat .env
GEMINI_API_KEY: AIzaSyBKpqc-AMWvzd_b9vaGZIqslr5j6XB5FLg

$ cat GEMINI.md
假设你是一名精通 LLM 相关的高级研究人员，并且也是一位 python 高级开发工程师，现在有一个专门用于 LLM 微调的项目 LLaMA-Factory。
请读取指定目录中的代码，分析代码逻辑，为代码中的核心逻辑添加详细的注释。
注意，您添加的注释要说明代码为什么要这么写，要解决的问题是什么。
$
$ gemini

 ███            █████████  ██████████ ██████   ██████ █████ ██████   █████ █████
░░░███         ███░░░░░███░░███░░░░░█░░██████ ██████ ░░███ ░░██████ ░░███ ░░███
  ░░░███      ███     ░░░  ░███  █ ░  ░███░█████░███  ░███  ░███░███ ░███  ░███
    ░░░███   ░███          ░██████    ░███░░███ ░███  ░███  ░███░░███░███  ░███
     ███░    ░███    █████ ░███░░█    ░███ ░░░  ░███  ░███  ░███ ░░██████  ░███
   ███░      ░░███  ░░███  ░███ ░   █ ░███      ░███  ░███  ░███  ░░█████  ░███
 ███░         ░░█████████  ██████████ █████     █████ █████ █████  ░░█████ █████
░░░            ░░░░░░░░░  ░░░░░░░░░░ ░░░░░     ░░░░░ ░░░░░ ░░░░░    ░░░░░ ░░░░░

Tips for getting started:
1. Ask questions, edit files, or run commands.
2. Be specific for the best results.
3. Create GEMINI.md files to customize your interactions with Gemini.
4. /help for more information.


╭───────────────────────────────────────────────────────────────╮
│ >   Type your message or @path/to/file  │
╭───────────────────────────────────────────────────────────────╮
 /workspaces/studio-release (main*)   no sandbox (see /docs)   auto




> /help

╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                              │
│ Basics:                                                                                                      │
│ Add context: Use @ to specify files for context (e.g., @src/myFile.ts) to target specific files or folders.  │
│ Shell mode: Execute shell commands via ! (e.g., !npm run start) or use natural language (e.g. start server). │
│                                                                                                              │
│ Commands:                                                                                                    │
│  /about - Show version info                                                                                  │
│  /auth - Change the auth method                                                                              │
│  /bug - Submit a bug report                                                                                  │
│  /chat - Manage conversation history                                                                         │
│    list - List saved conversation checkpoints                                                                │
│    save - Save the current conversation as a checkpoint. Usage: /chat save <tag>                             │
│    resume - Resume a conversation from a checkpoint. Usage: /chat resume <tag>                               │
│    delete - Delete a conversation checkpoint. Usage: /chat delete <tag>                                      │
│    share - Share the current conversation to a markdown or json file. Usage: /chat share <file>              │
│  /clear - Clear the screen and conversation history                                                          │
│  /compress - Compresses the context by replacing it with a summary                                           │
│  /copy - Copy the last result or code snippet to clipboard                                                   │
│  /docs - Open full Gemini CLI documentation in your browser                                                  │
│  /directory - Manage workspace directories                                                                   │
│    add - Add directories to the workspace. Use comma to separate multiple paths                              │
│    show - Show all directories in the workspace                                                              │
│  /editor - Set external editor preference                                                                    │
│  /extensions - Manage extensions                                                                             │
│    list - List active extensions                                                                             │
│    update - Update extensions. Usage: update <extension-names>|--all                                         │
│    explore - Open extensions page in your browser                                                            │
│  /help - For help on gemini-cli                                                                              │
│  /ide - Manage IDE integration                                                                               │
│    enable - Enable IDE integration                                                                           │
│    status - Check status of IDE integration                                                                  │
│    install - Install required IDE companion for GitHub Codespaces                                            │
│  /init - Analyzes the project and creates a tailored GEMINI.md file                                          │
│  /mcp - Manage configured Model Context Protocol (MCP) servers                                               │
│    list - List configured MCP servers and tools                                                              │
│    desc - List configured MCP servers and tools with descriptions                                            │
│    schema - List configured MCP servers and tools with descriptions and schemas                              │
│    auth - Authenticate with an OAuth-enabled MCP server                                                      │
│    refresh - Restarts MCP servers                                                                            │
│  /memory - Commands for interacting with memory                                                              │
│    show - Show the current memory contents                                                                   │
│    add - Add content to the memory                                                                           │
│    refresh - Refresh the memory from the source                                                              │
│    list - Lists the paths of the GEMINI.md files in use                                                      │
│  /model - Opens a dialog to configure the model                                                              │
│  /privacy - Display the privacy notice                                                                       │
│  /quit - Exit the cli                                                                                        │
│  /stats - Check session stats. Usage: /stats [model|tools]                                                   │
│    model - Show model-specific usage statistics                                                              │
│    tools - Show tool-specific usage statistics                                                               │
│  /theme - Change the theme                                                                                   │
│  /tools - List available Gemini CLI tools. Usage: /tools [desc]                                              │
│  /settings - View and edit Gemini CLI settings                                                               │
│  /vim - Toggle vim mode on/off                                                                               │
│  /setup-github - Set up GitHub Actions                                                                       │
│  /terminal-setup - Configure terminal keybindings for multiline input (VS Code, Cursor, Windsurf)            │
│  ! - shell command                                                                                           │
│ [MCP] - Model Context Protocol command (from external servers)                                               │
│                                                                                                              │
│ Keyboard Shortcuts:                                                                                          │
│ Alt+Left/Right - Jump through words in the input                                                             │
│ Ctrl+C - Quit application                                                                                    │
│ Ctrl+J - New line (Alt+Enter works for certain linux distros)                                                │
│ Ctrl+L - Clear the screen                                                                                    │
│ Ctrl+X - Open input in external editor                                                                       │
│ Ctrl+Y - Toggle YOLO mode                                                                                    │
│ Enter - Send message                                                                                         │
│ Esc - Cancel operation / Clear input (double press)                                                          │
│ Shift+Tab - Toggle auto-accepting edits                                                                      │
│ Up/Down - Cycle through your prompt history                                                                  │
│                                                                                                              │
│ For a full list of shortcuts, see docs/cli/keyboard-shortcuts.md   


```
