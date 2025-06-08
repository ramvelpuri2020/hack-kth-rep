import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'HandSync - Sign Language Practice',
  description: 'Real-time video calling platform for sign language practice sessions',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
