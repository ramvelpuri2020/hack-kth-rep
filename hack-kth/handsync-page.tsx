"use client"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Clock, Users, Snowflake } from "lucide-react"

export default function HandSyncPage() {
  const [durationFilters, setDurationFilters] = useState({
    "0-1 hours": false,
    "2-3 hours": false,
    "4-5 hours": false,
    "5+ hours": false,
  })

  const [difficultyFilters, setDifficultyFilters] = useState({
    Beginner: false,
    Learning: false,
    Associate: false,
    Proficient: false,
    Advanced: false,
  })

  const groups = [
    {
      title: "Learning Group",
      hours: 2,
      people: 3,
      tag: null,
    },
    {
      title: "Let's Get Better",
      hours: 2,
      people: 4,
      tag: null,
    },
    {
      title: "Group Study",
      hours: 3,
      people: 7,
      tag: "BEGINNER",
    },
    {
      title: "Training",
      hours: 4,
      people: 5,
      tag: "ADVANCED",
    },
    {
      title: "Group Study",
      hours: 2,
      people: 15,
      tag: null,
    },
  ]

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-teal-500 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-2 text-white">
          <Snowflake className="w-6 h-6" />
          <span className="text-xl font-medium">HandSync</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-white text-sm">Sort by</span>
          <Select defaultValue="people">
            <SelectTrigger className="w-32 bg-white">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="people"># of People</SelectItem>
              <SelectItem value="duration">Duration</SelectItem>
              <SelectItem value="difficulty">Difficulty</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar */}
        <aside className="w-64 p-6 bg-white border-r">
          {/* Duration Filter */}
          <div className="mb-8">
            <h3 className="font-medium text-gray-900 mb-4 flex items-center justify-between">
              Duration
              <span className="text-gray-400">▼</span>
            </h3>
            <div className="space-y-3">
              {Object.entries(durationFilters).map(([duration, checked]) => (
                <div key={duration} className="flex items-center space-x-2">
                  <Checkbox
                    id={duration}
                    checked={checked}
                    onCheckedChange={(checked) =>
                      setDurationFilters((prev) => ({
                        ...prev,
                        [duration]: checked as boolean,
                      }))
                    }
                  />
                  <label htmlFor={duration} className="text-sm text-gray-700 cursor-pointer">
                    {duration}
                  </label>
                </div>
              ))}
            </div>
          </div>

          {/* Difficulty Filter */}
          <div>
            <h3 className="font-medium text-gray-900 mb-4 flex items-center justify-between">
              Difficulty
              <span className="text-gray-400">▼</span>
            </h3>
            <div className="space-y-3">
              {Object.entries(difficultyFilters).map(([difficulty, checked]) => (
                <div key={difficulty} className="flex items-center space-x-2">
                  <Checkbox
                    id={difficulty}
                    checked={checked}
                    onCheckedChange={(checked) =>
                      setDifficultyFilters((prev) => ({
                        ...prev,
                        [difficulty]: checked as boolean,
                      }))
                    }
                  />
                  <label htmlFor={difficulty} className="text-sm text-gray-700 cursor-pointer">
                    {difficulty}
                  </label>
                </div>
              ))}
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 p-6">
          <div className="space-y-4 max-w-2xl">
            {groups.map((group, index) => (
              <Card key={index} className="border border-gray-200 shadow-sm">
                <CardContent className="p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      {group.tag && (
                        <div className="mb-2">
                          <span
                            className={`inline-block px-2 py-1 text-xs font-medium rounded ${
                              group.tag === "BEGINNER" ? "bg-orange-100 text-orange-800" : "bg-red-100 text-red-800"
                            }`}
                          >
                            {group.tag}
                          </span>
                        </div>
                      )}
                      <h3 className="font-medium text-gray-900 mb-3">{group.title}</h3>
                      <div className="flex items-center gap-4 text-sm text-gray-600">
                        <div className="flex items-center gap-1">
                          <Clock className="w-4 h-4" />
                          <span>{group.hours} hours</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Users className="w-4 h-4" />
                          <span>{group.people} People</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}

            {/* Load More Button */}
            <div className="pt-4">
              <Button variant="outline" className="w-full text-teal-600 border-teal-200 hover:bg-teal-50">
                Load More
              </Button>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
