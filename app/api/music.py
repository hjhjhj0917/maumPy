from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.music import get_music_recommendations

router = APIRouter()


class MusicRecommendRequest(BaseModel):
    mainEmotion: str


class TrackDTO(BaseModel):
    trackId: str
    trackName: str
    artistName: str
    albumImageUrl: Optional[str] = None
    spotifyUrl: Optional[str] = None


class MusicRecommendResponse(BaseModel):
    tracks: List[TrackDTO] = []


@router.post("/api/music/recommend", response_model=MusicRecommendResponse)
async def recommend_music(req: MusicRecommendRequest):
    results = get_music_recommendations(req.mainEmotion)

    tracks = [
        TrackDTO(
            trackId=t["track_id"],
            trackName=t["track_name"],
            artistName=t["artist_name"],
            albumImageUrl=t["album_image_url"],
            spotifyUrl=t["spotify_url"],
        )
        for t in results
    ]

    return MusicRecommendResponse(tracks=tracks)
