"use client"

import { Snowflake } from "lucide-react"
import Link from "next/link"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"

export default function LoginPage() {
  return (
    <div className="min-h-screen flex">
      {/* Left side */}
      <div className="w-2/5 bg-gradient-to-br from-teal-400 to-teal-600 p-12 flex flex-col">
        <div className="flex items-center gap-3 text-white mb-16">
          <Snowflake className="w-8 h-8" />
          <span className="text-2xl font-medium">HandSync</span>
        </div>

        <div className="text-white mt-8">
          <h1 className="text-4xl font-light mb-2">Sign anywhere,</h1>
          <h1 className="text-4xl font-bold mb-12">Anytime.</h1>

          <p className="text-lg">
            We offer <span className="font-bold">peer-to-peer</span> sign language exchange with{" "}
            <span className="font-bold">real-time translation</span>. Connect with others though our website.
          </p>
        </div>

        {/* Hand signs illustration */}
        <div className="flex items-center justify-center gap-4 mt-auto">
          <div className="w-16 h-16 bg-brown-500 rounded-full"></div>
          <div className="w-16 h-16 bg-orange-300 rounded-full"></div>
          <div className="w-16 h-16 bg-brown-700 rounded-full"></div>
          <div className="w-16 h-16 bg-orange-400 rounded-full"></div>
        </div>
      </div>

      {/* Right side */}
      <div className="w-3/5 bg-white p-12 flex flex-col items-center justify-center">
        <div className="w-full max-w-md">
          <h2 className="text-3xl font-medium text-gray-800 mb-8 text-center">Welcome Back!</h2>

          {/* Social login buttons */}
          <div className="grid grid-cols-2 gap-4 mb-8">
            <button className="flex items-center justify-center gap-2 border border-gray-300 rounded-md py-3 px-4 hover:bg-gray-50 transition-colors">
              <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none">
                <path
                  d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                  fill="#4285F4"
                />
                <path
                  d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                  fill="#34A853"
                />
                <path
                  d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                  fill="#FBBC05"
                />
                <path
                  d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                  fill="#EA4335"
                />
              </svg>
              <span className="text-gray-600">Continue with Google</span>
            </button>
            <button className="flex items-center justify-center gap-2 border border-gray-300 rounded-md py-3 px-4 hover:bg-gray-50 transition-colors">
              <div className="w-6 h-6 bg-blue-600 rounded flex items-center justify-center text-white font-bold">C</div>
              <span className="text-gray-600">Continue with Clever</span>
            </button>
          </div>

          {/* Divider */}
          <div className="flex items-center justify-center gap-4 mb-8">
            <div className="h-px bg-gray-300 flex-1"></div>
            <span className="text-gray-400 text-sm">- OR -</span>
            <div className="h-px bg-gray-300 flex-1"></div>
          </div>

          {/* Login form */}
          <form className="space-y-6">
            <div>
              <Input
                type="email"
                placeholder="Email Address"
                className="border-0 border-b border-gray-300 rounded-none px-0 py-2 focus-visible:ring-0 focus-visible:border-gray-500"
              />
            </div>
            <div>
              <Input
                type="password"
                placeholder="Password"
                className="border-0 border-b border-gray-300 rounded-none px-0 py-2 focus-visible:ring-0 focus-visible:border-gray-500"
              />
            </div>
            <Button className="w-full py-6 bg-blue-600 hover:bg-blue-700 text-lg">Login</Button>
          </form>

          <div className="text-center mt-6">
            <Link href="/signup" className="text-gray-500 hover:text-gray-700">
              New here? Sign up here
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}
