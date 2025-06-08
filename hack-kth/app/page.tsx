import Link from 'next/link';

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-400 to-teal-600 flex items-center justify-center">
      <div className="text-center text-white">
        <div className="mb-8">
          <h1 className="text-6xl font-bold mb-4">HandSync</h1>
          <p className="text-2xl opacity-90">Practice before your session!</p>
        </div>
        
        <div className="space-y-4">
          <Link href="/video-call">
            <button className="block mx-auto px-12 py-4 bg-orange-300 hover:bg-orange-400 text-white text-xl font-medium rounded-full transition-colors">
              Create a Room
            </button>
          </Link>
          <Link href="/video-call">
            <button className="block mx-auto px-12 py-4 bg-orange-300 hover:bg-orange-400 text-white text-xl font-medium rounded-full transition-colors">
              Join a Room
            </button>
          </Link>
        </div>
        
        <div className="mt-16">
          <p className="text-lg opacity-75">WebRTC-powered video calling for HandSync practice sessions</p>
        </div>
      </div>
    </div>
  )
}
