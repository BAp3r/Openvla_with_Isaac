"""
终端输入处理
"""

import threading


class InputHandler:
    """处理终端输入"""
    
    def __init__(self, instructions: dict = None):
        self.current_command = None
        self.running = True
        self.instructions = instructions or {
            "抓取香蕉": "pick up the banana",
            "抓取苹果": "pick up the apple",
            "把香蕉放到苹果旁边": "place the banana next to the apple",
        }
        self._start_input_thread()
    
    def _start_input_thread(self):
        """启动输入监听线程"""
        def input_loop():
            self._print_help()
            
            while self.running:
                try:
                    cmd = input("\n请输入指令 >>> ").strip()
                    if cmd:
                        self.current_command = cmd
                except EOFError:
                    break
                except Exception as e:
                    print(f"输入错误: {e}")
        
        thread = threading.Thread(target=input_loop, daemon=True)
        thread.start()
    
    def _print_help(self):
        """打印帮助信息"""
        print("\n" + "=" * 60)
        print("OpenVLA + Isaac Sim 抓取演示")
        print("=" * 60)
        print("可用指令:")
        for cn, en in self.instructions.items():
            print(f"  - {cn} / {en}")
        print("  - stop: 停止当前动作")
        print("  - quit/exit: 退出程序")
        print("=" * 60)
    
    def get_command(self) -> str:
        """获取并清除当前命令"""
        cmd = self.current_command
        self.current_command = None
        return cmd
    
    def translate_command(self, cmd: str) -> str:
        """翻译中文指令为英文"""
        cmd_lower = cmd.lower()
        
        # 特殊命令
        if cmd_lower in ["quit", "exit", "q"]:
            return "QUIT"
        elif cmd_lower == "stop":
            return "STOP"
        
        # 翻译中文指令
        for cn, en in self.instructions.items():
            if cn in cmd:
                return en
        
        return cmd
    
    def stop(self):
        """停止输入处理"""
        self.running = False
