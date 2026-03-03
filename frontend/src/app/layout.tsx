import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'InterVo - Mathematics Admission Interview',
  description: 'AI-powered professional mathematics interviewer for SST admissions.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
