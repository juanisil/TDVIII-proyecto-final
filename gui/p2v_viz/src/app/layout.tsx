import "~/styles/globals.css";

import { GeistSans } from "geist/font/sans";
import { type Metadata } from "next";

export const metadata: Metadata = {
  title: "Visualización Embeddings Player2Vec - EPL 12/13",
  description: "Visualización de Embeddings de Player2Vec sobre Jugadores de la English Premier League 2012/13. Cómo encontrar al mejor jugador para tu equipo de fútbol.",
  icons: [{ rel: "icon", url: "/favicon.ico" }],
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${GeistSans.variable}`}>
      <body>{children}</body>
    </html>
  );
}
