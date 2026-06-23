import "./globals.css";

export const metadata = {
  title: "RAG Agent Pro",
  description: "检索增强智能体教学版"
};

export default function RootLayout({ children }) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
