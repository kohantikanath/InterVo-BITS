import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'InterVo - Hiring-Grade DSA Interview',
  description: 'Structured AI interview sessions and recruiter scorecards for DSA hiring.',
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
