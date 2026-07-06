import "./globals.css";

export const metadata = {
  title: "agent for train",
  description: "AI Agent 教学与调试应用"
};

export default function RootLayout({ children }) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
